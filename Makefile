# HypPAR — UPAR Track 2 (WACV)
# Run from repo root on Linux after: export PYTHONPATH=$PWD
# Point data: export HYPPAR_UPAR_ROOT=/path/to/UPAR-Challenge-2027

PYTHON ?= python3
CONFIG ?= configs/main_lorentz_entail.env
-include $(CONFIG)

export PYTHONPATH := $(CURDIR)$(if $(PYTHONPATH),:$(PYTHONPATH),)

.PHONY: help setup smoke data train-main train-euclid train-small train-paper \
        eval-main eval-modes package clean-ckpts curriculum

help:
	@echo "HypPAR Makefile targets:"
	@echo "  make setup          install Python deps (incl. gdown)"
	@echo "  make data           clone UPAR-Challenge-2027 + download Market/PA100K/PETA"
	@echo "  make smoke          quick import/dataset check"
	@echo "  make train-main     clean curriculum: Lorentz entailment (Tiny)"
	@echo "  make train-euclid   clean Euclid twin (same schedule)"
	@echo "  make train-small    clean Lorentz entailment (ConvNeXt-Small)"
	@echo "  make train-paper    main + euclid twin"
	@echo "  make eval-main      eval entailment on val"
	@echo "  make eval-modes     eval entailment/distance/attr_l1/hybrid"
	@echo "  make package        build Codabench zip from main ckpt"
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

curriculum:
	$(CURRICULUM)

train-paper: train-main train-euclid

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
