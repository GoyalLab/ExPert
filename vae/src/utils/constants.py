from typing import NamedTuple


class _REGISTRY_KEYS_NT(NamedTuple):
    X_KEY: str = "X"
    B_KEY: str = "basal"
    ATAC_X_KEY: str = "atac"
    BATCH_KEY: str = "batch"
    SAMPLE_KEY: str = "sample"
    LABELS_KEY: str = "labels"
    GENE_EMB_KEY: str = "gene_embedding"
    CTX_EMB_KEY: str = "ctx_embedding"
    CLS_EMB_KEY: str = "cls_embedding"
    CLS_SIM_KEY: str = "cls_embedding_sim"
    CLS_EMB_INIT: str = "CLS_EMB_INIT"
    CLS_CERT_KEY: str = "cls_certainty"
    PROTEIN_EXP_KEY: str = "proteins"
    CAT_COVS_KEY: str = "extra_categorical_covs"
    CONT_COVS_KEY: str = "extra_continuous_covs"
    INDICES_KEY: str = "ind_x"
    SIZE_FACTOR_KEY: str = "size_factor"
    MINIFY_TYPE_KEY: str = "minify_type"
    LATENT_QZM_KEY: str = "latent_qzm"
    LATENT_QZV_KEY: str = "latent_qzv"
    OBSERVED_LIB_SIZE: str = "observed_lib_size"
    GROUP_BASE_KEY: str = "group_"
    SPLIT_KEY: str = "split"


class _MODULE_KEYS(NamedTuple):
    X_KEY: str = "x"
    B_KEY: str = "b"
    G_EMB_KEY: str = "g"
    C_EMB_KEY: str = "e"
    # inference (rna)
    Z_KEY: str = "z"
    QZ_KEY: str = "qz"
    QZM_KEY: str = "qzm"
    QZV_KEY: str = "qzv"
    PWDX_KEY: str = "xpwd"
    # inference (latent to class embedding projection)
    ZG_KEY: str = "zg"
    QZG_KEY: str = "qzg"
    QZGM_KEY: str = "qzgm"
    QZGV_KEY: str = "qzgv"
    PWDG_KEY: str = "gpwd"
    # inference (class embedding to latent projection)
    ZE_KEY: str = "ze"
    LIBRARY_G_KEY: str = "library_g"
    Z_SHARED_KEY: str = "z_shared"
    CTX_LOGITS_KEY: str = "ctx_logits"
    CLS_LOGITS_KEY: str = "cls_logits"
    CTX_PROJ_KEY: str = "ctx_proj"
    CLS_PROJ_KEY: str = "cls_proj"
    # inference (library)
    LIBRARY_KEY: str = "library"
    LABEL_KEY: str = "label"
    QL_KEY: str = "ql"
    BATCH_INDEX_KEY: str = "batch_index"
    Y_KEY: str = "y"
    CONT_COVS_KEY: str = "cont_covs"
    CAT_COVS_KEY: str = "cat_covs"
    SIZE_FACTOR_KEY: str = "size_factor"
    ZT_G_KEY: str = "zt_g"
    QT_G_KEY: str = "qt_g"
    # generative
    PX_KEY: str = "px"
    PG_KEY: str = "pg"
    PL_KEY: str = "pl"
    PZ_KEY: str = "pz"
    PZG_KEY: str = "pzg"
    # loss
    KL_L_KEY: str = "kl_divergence_l"
    KL_Z_KEY: str = "kl_divergence_z"
    KL_Z_PER_LATENT_KEY: str = "kl_divergence_per_latent"
    KL_ZG_KEY: str = "kl_divergence_zg"
    R2_MEAN_KEY: str = "r2_mean"
    R2_VAR_KEY: str = "r2_var"
    # prediction
    PREDICTION_KEY: str = "cls_prediction"

class _TRAINING_KEYS_NT(NamedTuple):
    MODEL_KEY: str = "model"
    RESULTS_KEY: str = "results"
    LATENT_KEY: str = "latent"
    OUTPUT_KEY: str = "version_dir"
    TEST_KEY: str = "test_results"

class _LOSS_KEYS_NT(NamedTuple):
    # Loss keys
    LOSS: str = "loss"
    CLS_LOSS: str = "cls_loss"
    ALIGN_LOSS: str = "align_loss"
    ADV_LOSS: str = "adversial_context_loss"
    UNSCALED_LOSS: str = "unscaled_loss"
    CTX_CLS_LOSS: str = "ctx_loss"
    # Data keys
    DATA: str = "data"
    LOGITS: str = "logits"
    Y: str = "y"
    CZ: str = "cz"
    W: str = "W"
    Z2C: str = "z2c"
    C2Z: str = "c2z"

class _EXT_CLS_EMB_INIT_NT(NamedTuple):
    MODEL_KEY: str = "model"
    LABELS_KEY: str = "labels"
    N_TRAIN_LABELS_KEY: str = "n_train_labels"
    N_UNSEEN_LABELS_KEY: str = "n_unseen_labels"
    CTRL_CLASS_KEY: str = "ctrl_class"
    CTRL_CLASS_IDX_KEY: str = "ctrl_class_idx"

class _PREDICTION_KEYS_NT(NamedTuple):
    PREDICTION_KEY: str = "prediction"
    SOFT_PREDICTION_KEY: str = "soft_predictions"
    CTX_PREDICTION_KEY: str = "ctx_prediction"
    CTX_SOFT_PREDICTION_KEY: str = "ctx_soft_prediction"
    ZS_KEY: str = "zs"
    WS_KEY: str = "Ws"
    ALIGN_PREDICTION_KEY: str = "aligned_prediction"
    ALIGN_SOFT_PREDICTION_KEY: str = "aligned_soft_prediction"
    Z2C_KEY: str = "z2c"
    C2Z_KEY: str = "c2z"
    # Misc
    TOP_N_PREDICTION_KEY: str = "top_n_prediction"
    REPORT_KEY: str = "cls_report"
    SUMMARY_KEY: str = "cls_report_summary"


REGISTRY_KEYS = _REGISTRY_KEYS_NT()
MODULE_KEYS = _MODULE_KEYS()
TRAINING_KEYS = _TRAINING_KEYS_NT()
LOSS_KEYS = _LOSS_KEYS_NT()
PREDICTION_KEYS = _PREDICTION_KEYS_NT()
EXT_CLS_EMB_INIT = _EXT_CLS_EMB_INIT_NT()
