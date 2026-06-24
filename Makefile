# ==============================================================================
# Multi-Task Operator Learning for Control — Makefile
# ==============================================================================
#
# Usage:
#   make help                     Show all available targets
#   make docker-build             Build the Docker image
#   make all                      Reproduce all results (takes a long time)
#   make train ENV=p2p_cost       Train a single environment
#   make figure7                  Reproduce a specific figure
#
# Most targets can be run either natively or inside Docker:
#   make train ENV=p2p_cost                   # native
#   make docker-run CMD="make train ENV=p2p_cost"  # Docker
#
# ==============================================================================

# ---------- Configuration ----------
PYTHON       := python
DOCKER_IMAGE := setonet-reproducibility
DOCKER_GPU   := --gpus all
SEED         := 42
DEVICE       := cuda

# Directories
CONFIG_DIR   := configs
CHECKPOINT_DIR := checkpoints
DATA_DIR     := data
OUTPUT_DIR   := outputs
FIGURE_DIR   := $(OUTPUT_DIR)/figures
TABLE_DIR    := $(OUTPUT_DIR)/tables

# Environments
OCP_ENVS     := p2p_cost p2p_cost_small p2p_dynamics quadrotor obstacle
IMUJOCO_ENVS := hopper halfcheetah walker2d
ALL_ENVS     := $(OCP_ENVS) $(IMUJOCO_ENVS)

# Adaptation methods
ADAPT_METHODS := setonet_ft last_branch last_both full_branch
META_METHODS  := maml setonet_meta setonet_meta_full

# ---------- Docker ----------

.PHONY: docker-build docker-run

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-run:
	docker run $(DOCKER_GPU) --rm \
		-v $(PWD)/checkpoints:/app/checkpoints \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/outputs:/app/outputs \
		$(DOCKER_IMAGE) -c "$(CMD)"

# ---------- Data Generation ----------

.PHONY: data data-ocp data-imujoco

data: data-ocp

data-ocp: $(addprefix data-,$(OCP_ENVS))

data-%:
	$(PYTHON) src/envs/generate_data.py \
		--config $(CONFIG_DIR)/$*.yaml \
		--output $(DATA_DIR)/$* \
		--seed $(SEED)

# iMuJoCo: train SAC experts, then collect demonstrations
# (Not included — see README for iMuJoCo data source)
data-imujoco: $(addprefix data-,$(IMUJOCO_ENVS))

data-hopper data-halfcheetah data-walker2d: data-%:
	$(PYTHON) src/envs/imujoco/train_sac.py \
		--config $(CONFIG_DIR)/$*.yaml \
		--output $(DATA_DIR)/$*/sac_experts \
		--seed $(SEED)
	$(PYTHON) src/envs/imujoco/collect_demos.py \
		--config $(CONFIG_DIR)/$*.yaml \
		--experts $(DATA_DIR)/$*/sac_experts \
		--output $(DATA_DIR)/$* \
		--seed $(SEED)

# ---------- Training ----------

.PHONY: train train-all train-ocp train-imujoco train-meta train-maml

# Train a single environment: make train ENV=p2p_cost
train:
	$(PYTHON) src/training/train.py \
		--config $(CONFIG_DIR)/$(ENV).yaml \
		--data $(DATA_DIR)/$(ENV) \
		--output $(CHECKPOINT_DIR)/$(ENV)/pretrained \
		--seed $(SEED) \
		--device $(DEVICE)

train-all: train-ocp

train-ocp: $(addprefix train-env-,$(OCP_ENVS))
train-imujoco: $(addprefix train-env-,$(IMUJOCO_ENVS))

train-env-%:
	$(MAKE) train ENV=$*

# Meta-training: make train-meta ENV=p2p_dynamics VARIANT=meta_full
train-meta:
	$(PYTHON) src/training/meta_train.py \
		--config $(CONFIG_DIR)/$(ENV).yaml \
		--data $(DATA_DIR)/$(ENV) \
		--variant $(VARIANT) \
		--output $(CHECKPOINT_DIR)/$(ENV)/$(VARIANT) \
		--seed $(SEED) \
		--device $(DEVICE)

# MAML baseline training
train-maml:
	$(PYTHON) src/training/train_maml.py \
		--config $(CONFIG_DIR)/$(ENV).yaml \
		--data $(DATA_DIR)/$(ENV) \
		--output $(CHECKPOINT_DIR)/$(ENV)/maml \
		--seed $(SEED) \
		--device $(DEVICE)

# Task-conditioned MLP baseline training: make train-baseline ENV=p2p_cost
train-baseline:
	$(PYTHON) src/training/train_baseline.py \
		--config $(CONFIG_DIR)/$(ENV).yaml \
		--data $(DATA_DIR)/$(ENV) \
		--output $(CHECKPOINT_DIR)/$(ENV)/baseline \
		--seed $(SEED) \
		--device $(DEVICE)

# Baseline 2 (context-conditioned MLP) training: make train-baseline2 ENV=quadrotor
train-baseline2:
	$(PYTHON) src/training/train_baseline2.py \
		--config $(CONFIG_DIR)/$(ENV).yaml \
		--data $(DATA_DIR)/$(ENV) \
		--output $(CHECKPOINT_DIR)/$(ENV)/baseline2 \
		--seed $(SEED) \
		--device $(DEVICE)

# ---------- Evaluation ----------

.PHONY: evaluate

# Evaluate a single method: make evaluate ENV=p2p_cost METHOD=setonet_ft STEPS=25
evaluate:
	$(PYTHON) src/evaluation/evaluate.py \
		--config $(CONFIG_DIR)/$(ENV).yaml \
		--data $(DATA_DIR)/$(ENV) \
		--checkpoint $(CHECKPOINT_DIR)/$(ENV) \
		--method $(METHOD) \
		--steps $(STEPS) \
		--output $(OUTPUT_DIR)/results/$(ENV)_$(METHOD)_$(STEPS).json \
		--seed $(SEED)

# Zero-shot eval: MLP baseline vs pretrained SetONet (side by side).
# make eval-baseline ENV=p2p_cost
eval-baseline:
	$(PYTHON) src/evaluation/evaluate_baseline.py \
		--config $(CONFIG_DIR)/$(ENV).yaml \
		--data $(DATA_DIR)/$(ENV) \
		--checkpoint $(CHECKPOINT_DIR)/$(ENV)/baseline \
		--pretrained $(CHECKPOINT_DIR)/$(ENV)/pretrained \
		--baseline2 $(CHECKPOINT_DIR)/$(ENV)/baseline2 \
		--output $(OUTPUT_DIR)/results/$(ENV)/baseline_zeroshot.json \
		--seed $(SEED)

# ---------- Tables ----------

.PHONY: table2 table-fitting

# Table 2: Main adaptation results
table2:
	@echo "=== Generating Table 2: Adaptation Results ==="
	@for env in $(OCP_ENVS); do \
		for method in pretrained $(ADAPT_METHODS) $(META_METHODS); do \
			for steps in 0 1 25; do \
				$(MAKE) evaluate ENV=$$env METHOD=$$method STEPS=$$steps; \
			done; \
		done; \
	done
	$(PYTHON) src/plotting/make_table2.py \
		--results $(OUTPUT_DIR)/results \
		--output $(TABLE_DIR)/table2.tex

# Appendix: Operator fitting table (multiple seeds, script not included)
table-fitting:
	@echo "=== Generating Fitting Table ==="
	@for env in $(OCP_ENVS); do \
		for seed in 1 2 3 4 5 6 7 8 9 10; do \
			$(MAKE) train ENV=$$env SEED=$$seed; \
		done; \
	done
	$(PYTHON) src/plotting/make_table_fitting.py \
		--checkpoints $(CHECKPOINT_DIR) \
		--output $(TABLE_DIR)/table_fitting.tex

# ---------- Figures ----------

.PHONY: figures figure4 figure5 figure6 figure7 figure8 figure9 figure10 figure13

figures: figure4 figure5 figure6 figure7 figure8

# Figure 4: Operator fitting (predictions + rollouts)
figure4:
	$(PYTHON) src/plotting/plot_fitting.py \
		--config $(CONFIG_DIR) \
		--data $(DATA_DIR) \
		--checkpoints $(CHECKPOINT_DIR) \
		--output $(FIGURE_DIR)/figure4.pdf

# Figure 5: Task resolution invariance
figure5:
	$(PYTHON) src/evaluation/resolution.py \
		--config $(CONFIG_DIR) \
		--data $(DATA_DIR) \
		--checkpoints $(CHECKPOINT_DIR) \
		--output $(OUTPUT_DIR)/results/resolution
	$(PYTHON) src/plotting/plot_resolution.py \
		--results $(OUTPUT_DIR)/results/resolution \
		--output $(FIGURE_DIR)/figure5.pdf

# Figure 6: MAML vs SetONet scatter plots
figure6:
	$(PYTHON) src/plotting/plot_maml_scatter.py \
		--config $(CONFIG_DIR) \
		--data $(DATA_DIR) \
		--checkpoints $(CHECKPOINT_DIR) \
		--output $(FIGURE_DIR)/figure6.pdf

# Figure 7: Cost-based adaptation (P2P-Cost OOD + Obstacle)
figure7:
	$(PYTHON) src/adaptation/cost_adapt.py \
		--config $(CONFIG_DIR)/p2p_cost.yaml \
		--data $(DATA_DIR)/p2p_cost \
		--checkpoint $(CHECKPOINT_DIR)/p2p_cost/pretrained \
		--ood \
		--output $(OUTPUT_DIR)/results/cost_adapt_p2p_ood
	$(PYTHON) src/adaptation/cost_adapt.py \
		--config $(CONFIG_DIR)/obstacle.yaml \
		--data $(DATA_DIR)/obstacle \
		--checkpoint $(CHECKPOINT_DIR)/obstacle/pretrained \
		--output $(OUTPUT_DIR)/results/cost_adapt_obstacle
	$(PYTHON) src/plotting/plot_cost_adapt.py \
		--p2p-results $(OUTPUT_DIR)/results/cost_adapt_p2p_ood \
		--obstacle-results $(OUTPUT_DIR)/results/cost_adapt_obstacle \
		--output $(FIGURE_DIR)/figure7.pdf

# Figure 8: Quadrotor OOD meta-training
figure8:
	$(PYTHON) src/evaluation/evaluate_ood.py \
		--config $(CONFIG_DIR)/quadrotor.yaml \
		--data $(DATA_DIR)/quadrotor \
		--checkpoints $(CHECKPOINT_DIR)/quadrotor \
		--output $(OUTPUT_DIR)/results/quadrotor_ood
	$(PYTHON) src/plotting/plot_quad_ood.py \
		--results $(OUTPUT_DIR)/results/quadrotor_ood \
		--output $(FIGURE_DIR)/figure8.pdf

# Figure 9: HalfCheetah control predictions (requires iMuJoCo data)
figure9:
	$(PYTHON) src/plotting/plot_cheetah_ctrl.py \
		--config $(CONFIG_DIR)/halfcheetah.yaml \
		--data $(DATA_DIR)/halfcheetah \
		--checkpoints $(CHECKPOINT_DIR)/halfcheetah \
		--output $(FIGURE_DIR)/figure9.pdf

# Figure 10: HalfCheetah adaptation grid (requires iMuJoCo data)
figure10:
	@echo "=== Running HalfCheetah adaptation sweep ==="
	@for method in setonet_ft maml setonet_meta setonet_meta_full; do \
		for demos in 1 5 10 25; do \
			for steps in 1 5 10 25 50 100 200; do \
				for seed in 1 2 3 4 5; do \
					$(PYTHON) src/evaluation/evaluate.py \
						--config $(CONFIG_DIR)/halfcheetah.yaml \
						--data $(DATA_DIR)/halfcheetah \
						--checkpoint $(CHECKPOINT_DIR)/halfcheetah \
						--method $$method \
						--steps $$steps \
						--num-demos $$demos \
						--seed $$seed \
						--output $(OUTPUT_DIR)/results/cheetah_grid/$${method}_$${demos}_$${steps}_$${seed}.json; \
				done; \
			done; \
		done; \
	done
	$(PYTHON) src/plotting/plot_cheetah_grid.py \
		--results $(OUTPUT_DIR)/results/cheetah_grid \
		--output $(FIGURE_DIR)/figure10.pdf

# ---------- Full Reproduction ----------

.PHONY: all all-from-scratch

# Reproduce all results from pretrained checkpoints
all: table2 figures

# Full reproduction from scratch (very slow — trains everything first)
all-from-scratch: data train-all all

# ---------- Utilities ----------

.PHONY: help clean

help:
	@echo ""
	@echo "Multi-Task Operator Learning — Reproducibility Makefile"
	@echo "======================================================="
	@echo ""
	@echo "Docker:"
	@echo "  docker-build                Build Docker image"
	@echo "  docker-run CMD='...'        Run command inside Docker"
	@echo ""
	@echo "Data Generation:"
	@echo "  data                        Generate all datasets"
	@echo "  data-ocp                    Generate OCP environment datasets"
	@echo "  data-imujoco                Train SAC experts + collect demos"
	@echo ""
	@echo "Training:"
	@echo "  train ENV=<env>             Train SetONet on one environment"
	@echo "  train-all                   Train on all environments"
	@echo "  train-meta ENV=<env> VARIANT=<meta|meta_full>"
	@echo "                              Meta-train SetONet"
	@echo "  train-maml ENV=<env>        Train MAML baseline"
	@echo ""
	@echo "Evaluation:"
	@echo "  evaluate ENV=<e> METHOD=<m> STEPS=<s>"
	@echo "                              Evaluate one method"
	@echo ""
	@echo "Results:"
	@echo "  table2                      Reproduce Table 2"
	@echo "  figure4 ... figure8         Reproduce individual figures"
	@echo "  figures                     Reproduce all figures"
	@echo "  all                         Reproduce all (from checkpoints)"
	@echo "  all-from-scratch            Full reproduction (trains first)"
	@echo ""
	@echo "Environments: $(OCP_ENVS)"
	@echo "Methods:      pretrained $(ADAPT_METHODS) $(META_METHODS)"
	@echo ""

clean:
	rm -rf $(OUTPUT_DIR)/results $(OUTPUT_DIR)/figures $(OUTPUT_DIR)/tables

