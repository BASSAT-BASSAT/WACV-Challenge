# HypPAR — UPAR Track 2 (WACV)
# Run from repo root on Linux after: export PYTHONPATH=$PWD
# Point data: export HYPPAR_UPAR_ROOT=/path/to/UPAR-Challenge-2027

PYTHON ?= python3
CONFIG ?= configs/main_lorentz_entail.env
-include $(CONFIG)

export PYTHONPATH := $(CURDIR)$(if $(PYTHONPATH),:$(PYTHONPATH),)

.PHONY: help setup smoke data train-main train-euclid train-small train-base train-paper \
        eval-main eval-modes package clean-ckpts curriculum \
	ablations ablations-dry eval-ablations train-par eval-par package-par lodo-par

help:
	@echo "HypPAR Makefile targets:"
	@echo "  make setup          install Python deps (incl. gdown)"
	@echo "  make data           clone UPAR-Challenge-2027 + download Market/PA100K/PETA"
	@echo "  make smoke          quick import/dataset check"
	@echo "  make train-main     clean curriculum: Lorentz entailment (Tiny)"
	@echo "  make train-euclid   clean Euclid twin (same schedule)"
	@echo "  make train-small    clean Lorentz entailment (ConvNeXt-Small)"
	@echo "  make train-base     clean Lorentz hybrid (ConvNeXt-Base, 8GB settings)"
	@echo "  make train-paper    main + euclid twin"
	@echo "  make ablations      train and evaluate the full geometry/backbone matrix"
	@echo "  make ablations-dry  print the ablation commands without training"
	@echo "  make eval-ablations evaluate existing ablation checkpoints"
	@echo "  make eval-main      eval entailment on val"
	@echo "  make eval-modes     eval entailment/distance/attr_l1/hybrid"
	@echo "  make package        build Codabench zip from main ckpt"
	@echo "  make train-par      strong PAR classifier (ConvNeXt-Base, focal+EMA)"
	@echo "  make eval-par       calibrated weighted-L1 retrieval on val"
	@echo "  make package-par    Codabench zip for PAR submission"
	@echo "  make lodo-par       leave-one-domain-out model selection (3 folds)"
	@echo ""
	@echo "Data docs: DATA_SETUP.md"
	@echo "Override:  make train-main CONFIG=configs/main_lorentz_entail_small.env"
	@echo "Data path: export HYPPAR_UPAR_ROOT=/path/to/UPAR-Challenge-2027"

setup:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/setup_upar_data.py
	@echo "Then: export HYPPAR_UPAR_ROOT=$$(dirname $(CURDIR))/UPAR-Challenge-2027"

smoke:
	$(PYTHON) -m hyppar.scripts.smoke_dataset

define CURRICULUM
	$(PYTHON) -m hyppar.scripts.train_curriculum \
		--name $(NAME) \
		--geometry $(GEOMETRY) \
		--score-mode $(SCORE_MODE) \
		--backbone $(BACKBONE) \
		--stage-a-epochs $(STAGE_A_EPOCHS) \
		--stage-b-epochs $(STAGE_B_EPOCHS) \
		--batch-size-a $(BATCH_A) \
		--batch-size-b $(BATCH_B) \
		--lambda-attr $(LAMBDA_ATTR) \
		--lambda-rank $(LAMBDA_RANK) \
		--entail-beta $(ENTAIL_BETA) \
		--unfreeze-mode $(UNFREEZE_MODE)
endef

train-main: CONFIG=configs/main_lorentz_entail.env
train-main:
	$(MAKE) curriculum CONFIG=$(CONFIG)

train-euclid: CONFIG=configs/euclid_twin.env
train-euclid:
	$(MAKE) curriculum CONFIG=$(CONFIG)

train-small: CONFIG=configs/main_lorentz_entail_small.env
train-small:
	$(MAKE) curriculum CONFIG=$(CONFIG)

train-base:
	$(PYTHON) -m hyppar.scripts.train_curriculum \
		--name lorentz_hybrid_base \
		--geometry lorentz --score-mode hybrid --backbone convnext_base \
		--stage-a-epochs 4 --stage-b-epochs 8 \
		--batch-size-a 2 --batch-size-b 2 \
		--grad-accum-a 16 --grad-accum-b 16 \
		--lambda-attr 4 --lambda-rank 1 \
		--unfreeze-mode stages --unfreeze-stages 2 \
		--num-workers 2

curriculum:
	$(CURRICULUM)

train-paper: train-main train-euclid

ablations:
	$(PYTHON) -m hyppar.scripts.run_ablations --epochs-a 4 --epochs-b 8 --batch-size 2 --grad-accum 8 --package

ablations-dry:
	$(PYTHON) -m hyppar.scripts.run_ablations --dry-run

eval-ablations:
	$(PYTHON) -m hyppar.scripts.run_ablations --skip-train

eval-main:
	$(PYTHON) -m hyppar.evaluate \
		--ckpt checkpoints/$(or $(NAME),lorentz_entail_tiny)/model.pt \
		--score-mode entailment \
		--out checkpoints/$(or $(NAME),lorentz_entail_tiny)/eval_val_entailment.json \
		--batch-size 64 --num-workers 4

eval-modes:
	@CKPT=checkpoints/$(or $(NAME),lorentz_entail_tiny)/model.pt; \
	for m in entailment distance attr_l1 hybrid; do \
		$(PYTHON) -m hyppar.evaluate --ckpt $$CKPT --score-mode $$m \
			--out checkpoints/$(or $(NAME),lorentz_entail_tiny)/eval_val_$$m.json \
			--batch-size 64 --num-workers 4; \
	done

package:
	$(PYTHON) -m hyppar.scripts.package_submission \
		--ckpt checkpoints/$(or $(NAME),lorentz_entail_tiny)/model.pt \
		--out checkpoints/hyppar_task2_submission.zip
	@echo "Upload checkpoints/hyppar_task2_submission.zip to Codabench"

clean-ckpts:
	rm -rf checkpoints/*

train-par: CONFIG=configs/par_convnext_base.env
train-par:
	$(PYTHON) -m hyppar.scripts.train_par \
		--name $(NAME) --backbone $(BACKBONE) \
		--epochs $(EPOCHS) --batch-size $(BATCH) --lr $(LR) --amp

eval-par:
	$(PYTHON) -m hyppar.scripts.evaluate_par \
		--ckpt checkpoints/$(or $(NAME),par_convnext_base)/model.pt \
		--save-artifacts \
		--out checkpoints/$(or $(NAME),par_convnext_base)/eval_val_par.json

package-par:
	$(PYTHON) -m hyppar.scripts.package_par_submission \
		--ckpt checkpoints/$(or $(NAME),par_convnext_base)/model.pt \
		--out checkpoints/par_task2_submission.zip

lodo-par:
	$(PYTHON) -m hyppar.scripts.lodo_eval --backbone $(or $(BACKBONE),convnext_base)
