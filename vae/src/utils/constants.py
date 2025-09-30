from typing import NamedTuple


class _REGISTRY_KEYS_NT(NamedTuple):
    X_KEY: str = "X"
    B_KEY: str = "basal"
    ATAC_X_KEY: str = "atac"
    BATCH_KEY: str = "batch"
    SAMPLE_KEY: str = "sample"
    LABELS_KEY: str = "labels"
    GENE_EMB_KEY: str = "gene_embedding"
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
    # inference (gene embedding)
    ZG_KEY: str = "zg"
    QZG_KEY: str = "qzg"
    QZGM_KEY: str = "qzgm"
    QZGV_KEY: str = "qzgv"
    PWDG_KEY: str = "gpwd"
    LIBRARY_G_KEY: str = "library_g"
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

REGISTRY_KEYS = _REGISTRY_KEYS_NT()
MODULE_KEYS = _MODULE_KEYS()
TRAINING_KEYS = _TRAINING_KEYS_NT()
