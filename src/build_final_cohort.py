from pathlib import Path

import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MRI_FILE = RAW_DATA_DIR / "mri_patients.csv"
GDC_SAMPLE_SHEET_FILE = RAW_DATA_DIR / "gdc_sample_sheet.tsv"
METADATA_FILE = RAW_DATA_DIR / "metadata.json"
CLINICAL_PATIENT_FILE = RAW_DATA_DIR / "data_clinical_patient.txt"
CLINICAL_SAMPLE_FILE = RAW_DATA_DIR / "data_clinical_sample.txt"

OUTPUT_FILE = PROCESSED_DATA_DIR / "final_cohort.csv"
EXCLUDED_FILE = PROCESSED_DATA_DIR / "excluded_her2_cases.csv"
RNA_DUPLICATES_FILE = PROCESSED_DATA_DIR / "rna_duplicate_samples.csv"

# Helpers
MISSING_VALUES = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "unknown",
    "[not available]",
    "[not applicable]",
    "[not evaluated]",
}


def normalize_patient_id(value):
    """
    Normalize a TCGA patient identifier to the patient-level barcode:
    TCGA-XX-XXXX
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip().upper()

    if not value:
        return np.nan

    parts = value.split("-")

    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])

    return value


def normalize_status(value):
    """
    Normalize HER2-related categorical values.
    """
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in MISSING_VALUES:
        return None

    return value


# MRI
def load_mri_patients():
    print("\n[1/5] Loading MRI patients...")

    df = pd.read_csv(MRI_FILE)

    required_columns = {
        "PatientID",
        "Modality",
        "StudyInstanceUID",
        "SeriesInstanceUID",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required MRI columns: {sorted(missing)}"
        )

    # Keep only MR studies
    df = df[df["Modality"].astype(str).str.upper().eq("MR")].copy()

    df["patient_id"] = df["PatientID"].apply(normalize_patient_id)

    # Patient-level MRI summary
    patient_summary = (
        df.groupby("patient_id", as_index=False)
        .agg(
            mri_study_count=("StudyInstanceUID", "nunique"),
            mri_series_count=("SeriesInstanceUID", "nunique"),
        )
    )

    patient_summary["has_mri"] = True

    print(f"MRI series: {len(df)}")
    print(f"Unique MRI patients: {patient_summary['patient_id'].nunique()}")

    return patient_summary


# RNA-Seq
def load_rna_samples():
    print("\n[2/5] Loading GDC RNA-Seq sample sheet...")

    df = pd.read_csv(
        GDC_SAMPLE_SHEET_FILE,
        sep="\t",
        dtype=str,
    )

    required_columns = {
        "File ID",
        "File Name",
        "Data Category",
        "Data Type",
        "Project ID",
        "Case ID",
        "Sample ID",
        "Tissue Type",
        "Tumor Descriptor",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required RNA-Seq columns: {sorted(missing)}"
        )

    # Keep only the intended TCGA-BRCA primary tumor gene-expression files
    df = df[
        df["Project ID"].eq("TCGA-BRCA")
        & df["Data Category"].eq("Transcriptome Profiling")
        & df["Data Type"].eq("Gene Expression Quantification")
        & df["Tissue Type"].str.lower().eq("tumor")
        & df["Tumor Descriptor"].str.lower().eq("primary")
        ].copy()

    df["patient_id"] = df["Case ID"].apply(
        normalize_patient_id
    )

    # Count RNA-Seq files per patient
    patient_file_counts = (
        df.groupby("patient_id")
        .size()
        .rename("rna_file_count")
    )

    df = df.merge(
        patient_file_counts,
        on="patient_id",
        how="left",
    )

    duplicate_df = df[
        df["rna_file_count"] > 1
        ].copy()

    print(f"Primary tumor RNA-Seq files: {len(df)}")
    print(
        f"Unique RNA-Seq patients: "
        f"{df['patient_id'].nunique()}"
    )
    print(
        "Patients with multiple primary RNA-Seq files: "
        f"{duplicate_df['patient_id'].nunique()}"
    )

    # Save duplicates BEFORE selecting one file
    if not duplicate_df.empty:
        duplicate_columns = [
            "patient_id",
            "Case ID",
            "Sample ID",
            "File ID",
            "File Name",
            "Tissue Type",
            "Tumor Descriptor",
            "rna_file_count",
        ]

        duplicate_columns = [
            col for col in duplicate_columns
            if col in duplicate_df.columns
        ]

        duplicate_df[
            duplicate_columns
        ].sort_values(
            by=["patient_id", "Sample ID", "File ID"]
        ).to_csv(
            RNA_DUPLICATES_FILE,
            index=False,
        )

        print(
            "\nRNA duplicate QC file created:"
        )
        print(
            f"  {RNA_DUPLICATES_FILE}"
        )

    # Determine whether duplicates are actually different
    # biological samples or multiple files from same sample
    if not duplicate_df.empty:
        duplicate_summary = (
            duplicate_df
            .groupby("patient_id")
            .agg(
                file_count=("File ID", "nunique"),
                sample_count=("Sample ID", "nunique"),
            )
            .reset_index()
        )

        print(
            "\nDuplicate RNA-Seq patient summary:"
        )
        print(
            duplicate_summary.to_string(index=False)
        )

    # Deterministic selection rule
    #
    # Current rule:
    #   - Prefer one unique primary tumor sample.
    #   - If several files correspond to the same Sample ID,
    #     keep one deterministic file.
    #   - If several different Sample IDs exist for one patient,
    #     keep the lexicographically first Sample ID temporarily,
    #     but flag the patient for manual review.
    #
    # This rule makes the pipeline reproducible, but patients
    # with >1 distinct Sample ID must be checked before the
    # cohort is considered scientifically final.

    df = df.sort_values(
        by=[
            "patient_id",
            "Sample ID",
            "File ID",
        ]
    ).copy()

    selected = (
        df.drop_duplicates(
            subset="patient_id",
            keep="first",
        )
        [
            [
                "patient_id",
                "File ID",
                "File Name",
                "Sample ID",
                "Tissue Type",
                "Tumor Descriptor",
                "rna_file_count",
            ]
        ]
        .rename(
            columns={
                "File ID": "rna_file_id",
                "File Name": "rna_file_name",
                "Sample ID": "rna_sample_id",
                "Tissue Type": "rna_tissue_type",
                "Tumor Descriptor": "rna_tumor_descriptor",
            }
        )
    )

    selected["has_rna_seq"] = True

    # Mark whether patient originally had duplicates
    selected["rna_has_multiple_files"] = (
            selected["rna_file_count"] > 1
    )

    print(
        "\nRNA-Seq selection complete."
    )

    print(
        "Selected one RNA-Seq file per patient: "
        f"{len(selected)}"
    )

    return selected


# cBioPortal sample-level clinical data
def load_clinical_samples():
    print("\n[3/5] Loading clinical sample data...")

    df = pd.read_csv(
        CLINICAL_SAMPLE_FILE,
        sep="\t",
        comment="#",
        dtype=str,
    )

    required_columns = {
        "PATIENT_ID",
        "SAMPLE_ID",
        "SAMPLE_TYPE",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required clinical sample columns: {sorted(missing)}"
        )

    df["patient_id"] = df["PATIENT_ID"].apply(
        normalize_patient_id
    )

    primary = df[
        df["SAMPLE_TYPE"].str.lower().eq("primary")
    ].copy()

    primary_summary = (
        primary.groupby("patient_id", as_index=False)
        .agg(
            cbio_primary_sample_count=("SAMPLE_ID", "nunique")
        )
    )

    primary_summary["has_cbio_primary_sample"] = True

    print(
        "Patients with a cBioPortal primary tumor sample: "
        f"{primary_summary['patient_id'].nunique()}"
    )

    return primary_summary


# HER2 labeling
def derive_her2_label(row):
    """
    Conservative HER2 ground-truth rule

    Priority:
    1. If IHC and FISH are both definitive but conflict -> exclude
    2. If FISH is definitive:
       - use FISH for equivocal/indeterminate/missing IHC
       - if IHC agrees, use the same label
    3. If FISH is unavailable/non-definitive but IHC is definitive:
       - use IHC
    4. Otherwise -> unresolved/excluded

    Returns:
        HER2_status: 1 = positive, 0 = negative, NaN = excluded
        HER2_source
        HER2_reason
    """

    ihc = normalize_status(row.get("IHC_HER2"))
    fish = normalize_status(row.get("HER2_FISH_STATUS"))

    definitive = {"positive", "negative"}

    # Explicit disagreement between two definitive assays
    if (
            ihc in definitive
            and fish in definitive
            and ihc != fish
    ):
        return pd.Series(
            {
                "HER2_status": np.nan,
                "HER2_source": "conflict",
                "HER2_reason": (
                    f"IHC={ihc}; FISH={fish}"
                ),
            }
        )

    # Definitive FISH result
    if fish in definitive:
        label = 1 if fish == "positive" else 0

        if ihc in definitive:
            source = "IHC+FISH"
        else:
            source = "FISH"

        return pd.Series(
            {
                "HER2_status": label,
                "HER2_source": source,
                "HER2_reason": (
                    f"IHC={ihc or 'missing/non-definitive'}; "
                    f"FISH={fish}"
                ),
            }
        )

    # Definitive IHC result when FISH is absent/non-definitive
    if ihc in definitive:
        label = 1 if ihc == "positive" else 0

        return pd.Series(
            {
                "HER2_status": label,
                "HER2_source": "IHC",
                "HER2_reason": (
                    f"IHC={ihc}; "
                    f"FISH={fish or 'missing/non-definitive'}"
                ),
            }
        )

    return pd.Series(
        {
            "HER2_status": np.nan,
            "HER2_source": "unresolved",
            "HER2_reason": (
                f"IHC={ihc or 'missing/non-definitive'}; "
                f"FISH={fish or 'missing/non-definitive'}"
            ),
        }
    )


def load_clinical_patients():
    print("\n[4/5] Loading clinical patient data and deriving HER2 labels...")

    df = pd.read_csv(
        CLINICAL_PATIENT_FILE,
        sep="\t",
        comment="#",
        dtype=str,
    )

    required_columns = {
        "PATIENT_ID",
        "IHC_HER2",
        "HER2_FISH_STATUS",
        "AGE",
        "MENOPAUSE_STATUS",
        "AJCC_TUMOR_PATHOLOGIC_PT",
        "AJCC_NODES_PATHOLOGIC_PN",
        "AJCC_METASTASIS_PATHOLOGIC_PM",
        "AJCC_PATHOLOGIC_TUMOR_STAGE",
        "ER_STATUS_BY_IHC",
        "PR_STATUS_BY_IHC",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required clinical columns: {sorted(missing)}"
        )

    df["patient_id"] = df["PATIENT_ID"].apply(
        normalize_patient_id
    )

    her2_labels = df.apply(
        derive_her2_label,
        axis=1,
    )

    df = pd.concat(
        [df, her2_labels],
        axis=1,
    )

    clinical_columns = [
        "patient_id",
        "HER2_status",
        "HER2_source",
        "HER2_reason",

        # Preserve original HER2 assay values for traceability
        "IHC_HER2",
        "HER2_IHC_PERCENT_POSITIVE",
        "HER2_IHC_SCORE",
        "HER2_FISH_STATUS",
        "HER2_COPY_NUMBER",
        "CENT17_COPY_NUMBER",
        "HER2_CENT17_RATIO",

        # Candidate clinical model features
        "AGE",
        "MENOPAUSE_STATUS",
        "SEX",
        "HISTOLOGICAL_SUBTYPE",
        "HISTOLOGICAL_DIAGNOSIS",
        "AJCC_PATHOLOGIC_TUMOR_STAGE",
        "AJCC_TUMOR_PATHOLOGIC_PT",
        "AJCC_NODES_PATHOLOGIC_PN",
        "AJCC_METASTASIS_PATHOLOGIC_PM",
        "ER_STATUS_BY_IHC",
        "PR_STATUS_BY_IHC",
        "PRIMARY_SITE_PATIENT",
        "SITE_OF_TUMOR_TISSUE",
    ]

    # Keep only columns actually present
    clinical_columns = [
        col for col in clinical_columns
        if col in df.columns
    ]

    df = df[clinical_columns].copy()

    print(f"Clinical patients: {df['patient_id'].nunique()}")

    print("\nHER2 labels in full clinical dataset:")
    print(
        df["HER2_status"]
        .map({0.0: "Negative", 1.0: "Positive"})
        .fillna("Excluded")
        .value_counts()
    )

    return df


# Merge
def build_final_cohort(
        mri_df,
        rna_df,
        clinical_patient_df,
        clinical_sample_df,
):
    print("\n[5/5] Building final multimodal cohort...")

    # Start from MRI patients because MRI is the smallest modality
    cohort = mri_df.merge(
        clinical_patient_df,
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )

    print(
        "MRI + clinical: "
        f"{cohort['patient_id'].nunique()}"
    )

    cohort = cohort.merge(
        rna_df,
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )

    print(
        "MRI + clinical + RNA-Seq: "
        f"{cohort['patient_id'].nunique()}"
    )

    cohort = cohort.merge(
        clinical_sample_df,
        on="patient_id",
        how="left",
        validate="one_to_one",
    )

    # Save unresolved/conflicting HER2 cases separately
    excluded = cohort[
        cohort["HER2_status"].isna()
    ].copy()

    # Final cohort contains only definitive HER2 labels
    cohort = cohort[
        cohort["HER2_status"].notna()
    ].copy()

    # Check RNA-Seq duplicate samples inside the FINAL cohort
    final_rna_duplicates = cohort[
        cohort["rna_has_multiple_files"] == True
        ].copy()

    print(
        "\nPatients with multiple RNA-Seq files "
        "inside FINAL cohort: "
        f"{len(final_rna_duplicates)}"
    )

    if not final_rna_duplicates.empty:
        print(
            final_rna_duplicates[
                [
                    "patient_id",
                    "rna_sample_id",
                    "rna_file_id",
                    "rna_file_name",
                    "rna_file_count",
                ]
            ].to_string(index=False)
        )

    cohort["HER2_status"] = (
        cohort["HER2_status"].astype(int)
    )

    cohort["HER2_label"] = cohort["HER2_status"].map(
        {
            0: "Negative",
            1: "Positive",
        }
    )

    # Binary flags for future checks
    cohort["has_mri"] = True
    cohort["has_rna_seq"] = True
    cohort["has_valid_her2"] = True

    cohort = cohort.sort_values(
        by="patient_id"
    ).reset_index(drop=True)

    excluded = excluded.sort_values(
        by="patient_id"
    ).reset_index(drop=True)

    return cohort, excluded


# QC
def print_quality_report(cohort, excluded):
    print("\n" + "=" * 60)
    print("FINAL COHORT QUALITY REPORT")
    print("=" * 60)

    print(f"Final patients: {len(cohort)}")

    print("\nHER2 class distribution:")
    print(
        cohort["HER2_label"]
        .value_counts()
        .to_string()
    )

    print("\nHER2 label source:")
    print(
        cohort["HER2_source"]
        .value_counts()
        .to_string()
    )

    print(
        f"\nExcluded because HER2 was unresolved/conflicting: "
        f"{len(excluded)}"
    )

    if not excluded.empty:
        print(
            excluded[
                [
                    "patient_id",
                    "IHC_HER2",
                    "HER2_FISH_STATUS",
                    "HER2_source",
                    "HER2_reason",
                ]
            ].to_string(index=False)
        )

    print("\nMissing values in selected clinical fields:")

    columns_to_check = [
        "AGE",
        "MENOPAUSE_STATUS",
        "AJCC_PATHOLOGIC_TUMOR_STAGE",
        "AJCC_TUMOR_PATHOLOGIC_PT",
        "AJCC_NODES_PATHOLOGIC_PN",
        "AJCC_METASTASIS_PATHOLOGIC_PM",
        "ER_STATUS_BY_IHC",
        "PR_STATUS_BY_IHC",
    ]

    for column in columns_to_check:
        if column in cohort.columns:
            missing = (
                cohort[column]
                .isna()
                .sum()
            )

            unavailable = (
                cohort[column]
                .astype(str)
                .str.lower()
                .isin(MISSING_VALUES)
                .sum()
            )

            print(
                f"{column}: "
                f"{missing + unavailable}/{len(cohort)}"
            )


# Main
def main():
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mri_df = load_mri_patients()
    rna_df = load_rna_samples()
    clinical_sample_df = load_clinical_samples()
    clinical_patient_df = load_clinical_patients()

    cohort, excluded = build_final_cohort(
        mri_df=mri_df,
        rna_df=rna_df,
        clinical_patient_df=clinical_patient_df,
        clinical_sample_df=clinical_sample_df,
    )

    print_quality_report(
        cohort,
        excluded,
    )

    cohort.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    excluded.to_csv(
        EXCLUDED_FILE,
        index=False,
    )

    print("\nFiles created:")
    print(f"  {OUTPUT_FILE}")
    print(f"  {EXCLUDED_FILE}")


if __name__ == "__main__":
    main()
