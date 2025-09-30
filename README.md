# ShotTracker — Project Roadmap

**Last Updated:** 2025-09-29 (Local Time)

---

## Introduction

This roadmap documents the full vision, technical plan, and reasoning behind the **ShotTracker** project.  
I started this project as a learning tool to improve my SWE abilities, showing _what_ I will build and _why_ I built it this way.

---

## Table of Contents

1. [ Vision](#vision)
2. [ Guiding Principles](#guiding-principles)
   - [ Why MVP First](#why-mvp-first)
3. [ High-Level Phases & Milestones](#high-level-phases--milestones)
4. [ Target MVP v0.1](#target-mvp-v01-end-of-week-2)
5. [ Tech Stack (initial)](#tech-stack-initial)
6. [ Repository Structure](#repository-structure)
   - [ Why This Repository Structure](#why-this-repository-structure)
7. [ Data Schema (JSONL for each session)](#data-schema-jsonl-for-each-session)
8. [ End-to-End Pipeline v0.1 (CLI)](#end-to-end-pipeline-v01-cli)
9. [ Algorithms (v0.1)](#algorithms-v01)
10. [ Testing Strategy](#testing-strategy)
11. [ Week-1 Plan (Hands-on)](#week-1-plan-hands-on)

- [ Why We Use uv](#why-we-use-uv)

12. [ Documentation Website (parallel, 30–60 min)](#documentation-website-parallel-30–60-min)
13. [ Issue Board (starter backlog)](#issue-board-starter-backlog)
14. [ Next Steps After Week-1](#next-steps-after-week-1)
15. [ Acceptance Criteria Templates](#acceptance-criteria-templates)
16. [ Notes](#notes)
17. [ Change Log](#change-log)

---

## Vision

Build a basketball shot tracking system that logs attempts, classifies make/miss, estimates shot trajectory (arc, entry angle), and correlates body mechanics with outcomes to generate coaching insights. Ship as:

1. Desktop prototype → 2) Web dashboard → 3) Mobile app (on-device or edge inference).

---

## Guiding Principles

- **MVP first, iterate fast.** MVP means Minimum Viable Product — build the smallest, simplest version that proves the concept works, then quickly improve it based on feedback. Ship narrow, prove value, then add complexity.
- **Reproducibility & tests.** Every feature has a scriptable demo and tests.
- **Telemetry & data contracts.** All components produce clear, versioned outputs.
- **Document as you build.** Keep lightweight, running notes and changelogs.

---

### Why MVP First

The **Minimum Viable Product** approach keeps scope small, feedback fast, and risk low. By proving the core loop early—detecting ball & rim, segmenting attempts, and classifying make/miss—we validate that the problem is solvable in our constraints before investing in advanced features. This sequencing prevents wasted work, guides data collection toward what actually matters, and makes each subsequent layer (trajectory, pose, mobile) sit on a solid, measured foundation.

---

## High-Level Phases & Milestones

**Phase 0 — Product Definition (Day 1–2)**

- Use cases: solo workouts, team practices, small gym setups.
- Success metric v0.1: ≥90% correct make/miss classification on static gym angle.
- Success metric v0.2: arc estimate within ±3° vs. manual measurement.
- Success metric v0.3: entry angle within ±2°, center offset within ±10 cm.

**Phase 1 — Data Collection & Labeling (Week 1–2)**

- Capture short clips at 30/60 fps from a fixed angle.
- Label: shot start/end frames, rim bbox, ball bbox, make/miss.
- Storage: parquet/JSONL with schema (see Data Schema below).

**Phase 2 — Shot Event Detection (Week 2–3)**

- Detect ball + rim each frame; associate tracks; segment attempts (start→release→rim contact→end).
- Output: segments with timestamps.

**Phase 3 — Outcome Classification (Week 3–4)**

- Heuristics baseline (net/rim intersection, speed drop) → ML refinement.
- Metric: precision/recall by gym/video.

**Phase 4 — Trajectory & Arc Estimation (Week 4–6)**

- Camera calibration (single-view homography to court plane).
- Smooth ball center track; fit 2D parabola; estimate peak height, entry angle.

**Phase 5 — Pose & Mechanics (Week 6–8)**

- Pose keypoints (shoulders/elbows/hips/knees/ankles/wrists).
- Extract features: release angle, elbow flare, knee flexion timing, jump height, follow-through duration.
- Correlate vs. outcomes; generate simple insights.

**Phase 6 — Backend & Web Dashboard (Week 5–8)**

- FastAPI service + Postgres. Entities: Session, Attempt, Trajectory, PoseMetrics.
- Web: simple dashboard for sessions, charts, and video playback with overlays.

**Phase 7 — Mobile Prototype (Week 8+)**

- Record session, live counters, optional on-device inference.
- Sync to backend when online.

---

## Target MVP v0.1 (End of Week 2)

> ** MVP Goal:** Desktop app/script: load a video → detect ball/rim → segment attempts → classify make/miss → write JSONL + MP4 with overlays.  
> **Target Accuracy:** ≥90% make/miss on your test clips.

---

## Tech Stack (initial)

- **Language:** Python 3.10
- **CV/ML:** OpenCV, ultralytics/YOLO for ball/rim, a lightweight pose model later (e.g., YOLO-Pose/Mediapipe alternative)
- **Data:** NumPy, pandas, pyarrow (parquet), JSONL
- **Math/Signal:** SciPy (savgol filter), scikit-learn
- **Backend (Phase 6):** FastAPI, SQLModel / SQLAlchemy, PostgreSQL
- **Web:** Next.js (or simple SvelteKit) for dashboard; Chart.js/Recharts; HLS video
- **Packaging/Dev:** uv or venv, ruff/black, pytest, pre-commit, Docker (later)

---

## Repository Structure
