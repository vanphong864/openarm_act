#!/usr/bin/env python3
"""Nhận dạng bộ chấp hành từ dữ liệu thật rồi kiểm chứng bằng phát lại lệnh thật trong MuJoCo.

Bước 1 (không cần sim): target thật của motor suy ra từ effort, `target = q + (effort + kd q̇)/kp`. Quét vmax của bộ
  giới hạn tốc độ để target mô phỏng từ joint_commands khớp target suy ra; sau đó hồi quy effort ~ kp (target − q)
  − kd q̇ để có kp, kd. Mỗi tay fit trên episode tay đó đang thao tác (tay còn lại gần như đứng yên).
Bước 2 (MuJoCo): phát lại joint_commands của cả episode (mở vòng, chỉ đặt trạng thái đầu) với mô hình trên; tìm
  giảm chấn / ma sát / armature khớp trên episode train, báo cáo sai số qpos và effort trên episode test.

Ví dụ:
  identify_actuators.py --raw "/media/space/New Volume/dataset/2026-10-13" --out simulation/realdata/actuators_20261013.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import h5py
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openarm_sim.actuation import SIDES, RateLimitedTargets, apply_actuators, vmax16  # noqa: E402

ROBOT = Path(__file__).resolve().parent.parent / "third_party/openarm_mujoco/v1/openarm_bimanual.xml"
FPS = 50
VMAX_GRID = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
             2.2, 2.5, 3.0, 4.0]


def load(raw: Path):
    eps = {}
    for p in sorted(raw.glob("*.hdf5")):
        try:
            with h5py.File(p, "r") as f:
                e = {k.split("/")[-1]: f[k][()].astype(np.float64) for k in
                     ("joint_commands", "observations/qpos", "observations/qvel", "observations/effort")}
        except Exception as ex:  # noqa: BLE001 (file cắt cụt / rỗng)
            print(f"bỏ qua {p.name}: {str(ex)[:70]}")
            continue
        q = e["qpos"]
        e["arm"] = "left" if q[:, 7].min() < 0.035 else ("right" if q[:, 15].min() < 0.035 else None)
        eps[p.stem] = e
    return eps


def limiter(cmd, q0, vmax):
    out, cur, step = np.empty(len(cmd)), q0, vmax / FPS
    for i, c in enumerate(cmd):
        if np.isfinite(c):
            cur = cur + np.clip(c - cur, -step, step)
        out[i] = cur
    return out


def fit_offline(eps, train):
    """vmax, kp, kd cho từng khớp mỗi tay (chỉ episode tay đó thao tác)."""
    res = {}
    for side, base in (("left", 0), ("right", 8)):
        names = [n for n in train if eps[n]["arm"] == side]
        kp, kd, vm, r2 = [], [], [], []
        for j in range(7):
            J = base + j
            # kp, kd sơ bộ từ khung đứng yên & bám lệnh (bộ giới hạn không tác động)
            E = [eps[n] for n in names]
            err = np.concatenate([e["joint_commands"][:, J] - e["qpos"][:, J] for e in E])
            qv = np.concatenate([e["qvel"][:, J] for e in E])
            tau = np.concatenate([e["effort"][:, J] for e in E])
            ok = np.isfinite(err) & (np.abs(err) < 0.1)
            p0 = np.linalg.lstsq(np.c_[err[ok], -qv[ok], np.ones(ok.sum())], tau[ok], rcond=None)[0]
            k0, d0 = p0[0], max(p0[1], 0.05)
            best = None
            for v in VMAX_GRID:
                se = 0.0
                for e in E:
                    q = e["qpos"][:, J]
                    tgt = q + (e["effort"][:, J] + d0 * e["qvel"][:, J]) / k0
                    se += np.sum((limiter(e["joint_commands"][:, J], q[0], v) - tgt) ** 2)
                if best is None or se < best[0]:
                    best = (se, v)
            v = best[1]
            A, T = [], []
            for e in E:
                q = e["qpos"][:, J]
                A.append(np.c_[limiter(e["joint_commands"][:, J], q[0], v) - q, -e["qvel"][:, J],
                               np.ones(len(q))])
                T.append(e["effort"][:, J])
            A, T = np.concatenate(A), np.concatenate(T)
            p = np.linalg.lstsq(A, T, rcond=None)[0]
            kp.append(float(p[0]))
            kd.append(float(max(p[1], 0.0)))
            vm.append(float(v))
            r2.append(float(1 - np.var(T - A @ p) / np.var(T)))
        res[side] = {"kp": kp, "kd": kd, "vmax": vm, "effort_R2_train": r2}
    return res


class Replayer:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(ROBOT))
        self.m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.qadr = np.array([m.joint(f"openarm_{s}_joint{i}").qposadr[0] for s in SIDES for i in range(1, 8)])
        self.dadr = np.array([m.joint(f"openarm_{s}_joint{i}").dofadr[0] for s in SIDES for i in range(1, 8)])
        self.fadr = {s: [m.joint(f"openarm_{s}_finger_joint{i}").qposadr[0] for i in (1, 2)] for s in SIDES}
        self.act = np.array([m.actuator(f"{s}_joint{i}_ctrl").id for s in SIDES for i in range(1, 8)])
        self.fact = {s: [m.actuator(f"{s}_finger{i}_ctrl").id for i in (1, 2)] for s in SIDES}
        self.sub = round(1 / (FPS * m.opt.timestep))
        self.arm16 = np.array([*range(7), *range(8, 15)])

    def run(self, params, e):
        m, d = self.m, self.d
        apply_actuators(m, params)
        q, cmd = e["qpos"], e["joint_commands"]
        mujoco.mj_resetData(m, d)
        d.qpos[self.qadr] = q[0, self.arm16]
        d.qvel[self.dadr] = e["qvel"][0, self.arm16]
        for s, g in (("left", 7), ("right", 15)):
            d.qpos[self.fadr[s]] = q[0, g]
        tg = RateLimitedTargets(vmax16(params), q[0])
        dt = m.opt.timestep
        out_q, out_tau = np.empty((len(q), 14)), np.empty((len(q), 14))
        out_q[0], out_tau[0] = q[0, self.arm16], e["effort"][0, self.arm16]
        for t in range(len(q) - 1):
            tg.set_command(cmd[t])
            for _ in range(self.sub):
                target = tg.step(dt)
                d.ctrl[self.act] = target[self.arm16]
                for s, g in (("left", 7), ("right", 15)):
                    d.ctrl[self.fact[s]] = target[g]
                mujoco.mj_step(m, d)
            out_q[t + 1] = d.qpos[self.qadr]
            out_tau[t + 1] = d.actuator_force[self.act]
        return out_q, out_tau


def score(rep, params, eps, names):
    """RMS sai số qpos (rad) và effort (N·m) theo 14 khớp, chỉ tính tay đang thao tác trong episode."""
    se_q, se_t, cnt = np.zeros(14), np.zeros(14), np.zeros(14)
    arm16 = rep.arm16
    for n in names:
        e = eps[n]
        q, tau = rep.run(params, e)
        cols = np.arange(7) if e["arm"] == "left" else np.arange(7, 14)
        se_q[cols] += ((q - e["qpos"][:, arm16]) ** 2).sum(0)[cols]
        se_t[cols] += ((tau - e["effort"][:, arm16]) ** 2).sum(0)[cols]
        cnt[cols] += len(q)
    return np.sqrt(se_q / np.maximum(cnt, 1)), np.sqrt(se_t / np.maximum(cnt, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tune-episodes", type=int, default=8, help="số episode train mỗi tay dùng khi tìm ma sát")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    eps = load(args.raw)
    eps = {k: v for k, v in eps.items() if v["arm"]}
    rng = np.random.default_rng(args.seed)
    names = sorted(eps)
    test = sorted(rng.choice(names, max(4, len(names) // 5), replace=False).tolist())
    train = [n for n in names if n not in test]
    print(f"{len(eps)} episode: {len(train)} train, {len(test)} test")

    off = fit_offline(eps, train)
    for s in SIDES:
        print(s, {k: np.round(v, 3).tolist() for k, v in off[s].items()})

    rep = Replayer()
    m0 = rep.m
    base_damp = float(m0.dof_damping[rep.dadr[0]])
    base_fric = float(m0.dof_frictionloss[rep.dadr[0]])
    params = {s: {**off[s], "damping": [base_damp] * 7, "frictionloss": [base_fric] * 7, "armature": [0.0] * 7,
                  "force_limit": [40, 40, 27, 27, 7, 7, 7]} for s in SIDES}
    params["gripper"] = {"kp": 100.0, "kv": 5.0, "force_limit": 30.0, "vmax": 0.05}

    # Đường cơ sở: cấu hình cũ (scene_openarm.yaml, không giới hạn tốc độ, servo cũ)
    old = copy.deepcopy(params)
    for s in SIDES:
        old[s].update(kp=[42.5, 42.5, 25.5, 25.5, 24.0, 24.0, 24.0], kd=[2.04, 2.04, 1.02, 1.02, 0.1, 0.1, 0.1],
                      vmax=[100.0] * 7)
    tune = [n for s in SIDES for n in [x for x in train if eps[x]["arm"] == s][:args.tune_episodes]]
    q_old, t_old = score(rep, old, eps, test)
    q_new, t_new = score(rep, params, eps, test)
    print(f"test qpos RMS (rad) cũ {q_old.mean():.4f} → mới (chưa tinh chỉnh) {q_new.mean():.4f}; "
          f"effort RMS (N·m) {t_old.mean():.3f} → {t_new.mean():.3f}")

    # Bước 2: tìm toạ độ (coordinate search) giảm chấn, ma sát, armature theo nhóm khớp, chung hai tay.
    groups = {"shoulder": [0, 1], "upper": [2, 3], "wrist": [4, 5, 6]}
    grid = {"damping": [0.0, 0.1, 0.2, 0.4, 0.8, 1.2], "frictionloss": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
            "armature": [0.0, 0.01, 0.03, 0.06, 0.1]}

    def cost(p):
        qe, te = score(rep, p, eps, tune)
        return float(qe.mean() + 0.01 * te.mean())

    best_c = cost(params)
    for sweep in range(2):
        for key, values in grid.items():
            for g, idx in groups.items():
                for v in values:
                    trial = copy.deepcopy(params)
                    for s in SIDES:
                        for i in idx:
                            trial[s][key][i] = v
                    c = cost(trial)
                    if c < best_c - 1e-6:
                        best_c, params = c, trial
                print(f"  sweep {sweep} {key}/{g}: {[params['left'][key][i] for i in idx]} cost {best_c:.5f}",
                      flush=True)
    q_fin, t_fin = score(rep, params, eps, test)
    print(f"test qpos RMS (rad): cũ {q_old.mean():.4f} | mới {q_fin.mean():.4f}; effort RMS cũ {t_old.mean():.3f} | "
          f"mới {t_fin.mean():.3f}")
    names14 = [f"{s}_j{i}" for s in SIDES for i in range(1, 8)]
    report = {
        "source": str(args.raw), "train_episodes": train, "test_episodes": test,
        "params": params,
        "test_qpos_rms_rad": {"old_config": dict(zip(names14, q_old.round(5).tolist())),
                              "identified": dict(zip(names14, q_fin.round(5).tolist()))},
        "test_effort_rms_Nm": {"old_config": dict(zip(names14, t_old.round(4).tolist())),
                               "identified": dict(zip(names14, t_fin.round(4).tolist()))},
        "notes": ["Không bù trọng lực: effort ≈ mô-men trọng lực URDF (hệ số ≈ 1.0, tương quan 0.99 ở vai).",
                  "Target tiến về joint_commands với tốc độ tối đa vmax (rad/s) — bộ giới hạn phía robot.",
                  "Sai số chỉ tính tay đang thao tác; phát lại mở vòng cả episode, không có vật trong sim."],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print("ghi", args.out)


if __name__ == "__main__":
    main()
