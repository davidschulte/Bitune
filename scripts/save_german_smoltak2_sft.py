from datasets import load_dataset
from langdetect import detect, DetectorFactory

# Keep output deterministic
DetectorFactory.seed = 0

OUTPUT_PATH = "data/smoltalk_sft_german"


def is_german(row):
    messages = row.get("messages", [])
    if not messages:
        return False

    first_user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    return detect(first_user_msg) == "de"


def remove_think_passage(row):
    content = row["messages"][1]["content"]
    content_think_splits = content.split("</think>\n\n")
    if len(content_think_splits) < 2:
        return None

    row["messages"][1]["content"] = content_think_splits[1]

    return row


if __name__ == "__main__":
    ds = load_dataset(
        "HuggingFaceTB/smoltalk2",
        data_files="SFT/smoltalk_multilingual8_Qwen3_32B_think-*.parquet",
    )

    german_ds = ds.filter(is_german)
    print(f"Successfully loaded exactly {len(german_ds)} German examples.")

    german_ds_without_think = german_ds.map(remove_think_passage)
    german_ds_without_think = german_ds_without_think.filter(lambda x: x is not None)

    print(
        f"Successfully retained {len(german_ds_without_think)} German examples after removing think passages."
    )

    german_ds_without_think.save_to_disk(OUTPUT_PATH)
