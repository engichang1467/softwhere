SHELL := /usr/bin/env bash

PY ?= $(CURDIR)/.venv/bin/python
CUDA_VISIBLE_DEVICES ?= 0

.EXPORT_ALL_VARIABLES:

export PY
export CUDA_VISIBLE_DEVICES

.PHONY: help prepare all coverage e2e exp1 exp2 exp3 exp4 exp5 exp6 exp7

help:
	@echo "SoftWhere experiment runners"
	@echo
	@echo "Targets:"
	@echo "  make prepare   Clone/update repos, install uv env, and prepare data"
	@echo "  make exp1      Resolution-parity TokenLearner-SR"
	@echo "  make exp2      Selection-policy ablation"
	@echo "  make exp3      NMS robustness sweep for div1"
	@echo "  make exp4      Diversity and fovea-count sweeps"
	@echo "  make exp5      Main div0 head sanity check"
	@echo "  make exp6      Mini end-to-end CLS training"
	@echo "  make exp7      Mini E2E CLS head sanity check"
	@echo "  make coverage  Run experiments 1-5"
	@echo "  make e2e       Run experiments 6-7"
	@echo "  make all       Run experiments 1-7"
	@echo
	@echo "Examples:"
	@echo "  make prepare SKIP_ADE20K=1"
	@echo "  make exp1 CUDA_VISIBLE_DEVICES=0"
	@echo "  make exp1 SOFTWHERE_AUTO_PREPARE=1"
	@echo "  make exp6 STEPS=500"
	@echo "  make exp3 K_VALUES='16 72 128 136' NMS_DISTS='1 2 3 4'"

prepare:
	./experiment_env.sh prepare

all: exp1 exp2 exp3 exp4 exp5 exp6 exp7

coverage: exp1 exp2 exp3 exp4 exp5

e2e: exp6 exp7

exp1:
	./run_exp1_resolution_parity.sh

exp2:
	./run_exp2_selection_policy_ablation.sh

exp3:
	./run_exp3_nms_robustness_div1.sh

exp4:
	./run_exp4_diversity_and_s_sweeps.sh

exp5:
	./run_exp5_main_div0_sanity_check.sh

exp6:
	./run_exp6_mini_e2e_cls.sh

exp7:
	./run_exp7_e2e_head_sanity_check.sh
