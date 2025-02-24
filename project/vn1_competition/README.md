# The Fine-Tuned Moirai-Base Model Achieves 1st Place in the VN1 Challenge

We present a reproducible experiment where **Salesforce's Moirai** pretrained model, after simple fine-tuning, achieves **first place** in the [VN1 Forecasting - Accuracy Challenge](https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge/description).

One of the competition's key requirements was to use open-source solutions, and all code, training scripts, and data for the Moirai pretrained model have been open-sourced. The further fine-tuning of the model follows the approach in the codebase, requiring only minor modifications to parameters in the fine-tuning scripts.

The table below displays the official competition results, where Moirai-base outperformed all competitors to claim the top position. The final scores were averaged over 5 predictions.

| **Model**   | **Score**  |
| ----------- | ---------- |
| **Moirai-base** | **0.4629** |
| 1st         | 0.4637 |
| 2nd         | 0.4657     |
| 3rd         | 0.4758     |
| 4th | 0.4774 | 
| 5th | 0.4808 |

---

### [**VN1 Forecasting**](https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge/description)
Participants in this datathon are tasked with accurately forecasting future sales using historical sales and pricing data provided. The goal is to develop robust predictive models that can anticipate sales trends for various products across different clients and warehouses. Submissions will be evaluated based on their accuracy and bias against actual sales figures. The competition is structured into two phases.

#### Phase 1
In this phase participants will use the provided Phase 0 sales data to predict sales for Phase 1. This phase will last three weeks, during which there will be live leaderboard updates to track the progress and provide feedback on the predictions. At the end of Phase 1, participants will receive the actual sales data for this phase.

#### Phase 2
Using both Phase 0 and Phase 1 data, participants will predict sales for Phase 2. This second phase will last two weeks, but unlike Phase 1, there will be no leaderboard updates until the competition ends.

---

### [**Data Overview**](https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge/datasets)
#### The competition data consists of three phases:
Phase 0: Historical training data  
Phase 1: Additional training data  
Phase 2: Test data (used for evaluation)

#### Each data entry includes the following fields:
- Client: Client ID  
- Warehouse: Warehouse ID  
- Product: Product ID  
- Weekly sales data

---

### **How to Run**
To reproduce the experimental results, refer to this [blog](https://zhuanlan.zhihu.com/p/20755649808).

#### Instructions
1. Follow the instructions from the `uni2ts` library to create a virtual environment and install dependencies.
2. The `Makefile` provides the raw dataset required for use:
   ```bash
   make download_data
   ```
3. After replacing the directory path of the downloaded raw dataset, run `prepare_data.py` to obtain the preprocessed dataset.
4. Add the directory path of the processed dataset to the `.env` file:
   ```bash
   echo "CUSTOM_DATA_PATH=PATH_TO_SAVE" >> .env
   ```
5. Replace the variable `pretrained_model_name_or_path` in the configuration file with your own path, then run the following command to fine-tune the `Moirai-base` model:
   ```bash
   python -m cli.train -cp ../project/vn1_competition/fine_tune run_name=run1
   ```
6. Replace the weight file path in the `main.py` file under the `src` directory and run `main.py`.

---

### **References**

- Vandeput, Nicolas. “VN1 Forecasting - Accuracy Challenge.” DataSource.ai, DataSource, 3 Oct. 2024, [https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge/description](https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge/description)
- [Moirai Paper](https://arxiv.org/abs/2402.02592)