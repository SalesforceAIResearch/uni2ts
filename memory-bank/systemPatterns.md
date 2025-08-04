# System Patterns: Uni2TS Fine-Tuning

## Architecture
The system follows a modular architecture centered around the `uni2ts` library, Hydra for configuration, and PyTorch Lightning for training.

- **Data Layer**: Financial data is stored in a Parquet data lake with a hive-style partitioning scheme. A data preparation script using `polars` and `datasets` will transform this data into a format compatible with `uni2ts`.
- **Configuration Layer**: Hydra is used to manage all configurations. The main configuration file (`default.yaml`) composes model, data, and trainer configurations. This allows for flexible and overridable settings.
- **Modeling Layer**: The core model is a pre-trained Moirai transformer from the `uni2ts` library. Fine-tuning is performed on this model.
- **Training Layer**: PyTorch Lightning handles the training loop, including optimization, checkpointing, and logging.

## Key Technical Decisions
- **Use of Pre-trained Models**: We will leverage a pre-trained Moirai model to benefit from its general time series understanding and reduce training time.
- **Hugging Face `datasets`**: This library will be used to create and manage the datasets, ensuring compatibility with the `uni2ts` framework.
- **Polars for Data Processing**: `polars` is chosen for its high performance in handling large Parquet datasets.
- **Hydra for Configuration**: Hydra's composition and override capabilities are ideal for managing complex experimental setups.

## Component Relationships
```mermaid
graph TD
    A[Parquet Data Lake] --> B(Data Preparation Script);
    B --> C[Hugging Face Dataset];
    C --> D{Data Loader};
    D --> E[PyTorch Lightning Trainer];
    F[Model Config] --> E;
    G[Data Config] --> D;
    H[Trainer Config] --> E;
    I[Moirai Model] --> E;

    subgraph Configuration [Hydra]
        F
        G
        H
    end
