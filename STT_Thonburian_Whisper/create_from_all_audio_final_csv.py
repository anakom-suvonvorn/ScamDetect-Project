import pandas as pd

for type in ["train", "test"]:
    scam_large_no_filter = pd.read_csv(f"final_transcription_files/large_transcribed_no_filter_final_{type}.csv")
    scam_medium_no_filter = pd.read_csv(f"final_transcription_files/medium_transcribed_no_filter_final_{type}.csv")
    scam_medium_filtered = pd.read_csv(f"final_transcription_files/medium_filtered_transcribed_final_{type}.csv")
    normal_medium_noised = pd.read_csv(f"final_transcription_files/right_call-center_trimmed_and_noised_transcribed_final_{type}.csv")

    s1 = scam_large_no_filter.sample(frac=1.0, random_state=42)
    s2 = scam_medium_no_filter.sample(frac=1.0, random_state=42)
    s3 = scam_medium_filtered.sample(frac=1.0, random_state=42)
    s4 = normal_medium_noised.sample(frac=1.0, random_state=42)

    final_df = pd.concat([s1, s2, s3, s4], ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(final_df['label'].value_counts())

    final_df.to_csv(f"final_transcription_files/all_audio_final_no_file_overlap_all_{type}.csv", index=False)
    print(f"Saved results to: final_transcription_files/all_audio_final_no_file_overlap_all_{type}.csv")