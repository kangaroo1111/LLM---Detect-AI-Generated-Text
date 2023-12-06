import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

EXTERNAL_AI_GENERATED_DATA_SET = "daigt_external_dataset - daigt_external_dataset.csv"
TRAIN_ESSAYS_PATH = "train_essays_RDizzl3_seven_v1 - train_essays_RDizzl3_seven_v1.csv"
TEST_ESSAYS_PATH = 'test_essays - test_essays.csv'


def prepare_dataframe_from_csv(csv_file_name):
    print("Reading dataframe from csv")
    dataframe = pd.read_csv(csv_file_name)
    dataframe = dataframe.rename(columns={'generated': 'label'})

    ai_generated_data_source = "source_text"
    human_data_source = "text"

    dataframe = dataframe[[ai_generated_data_source]]
    dataframe.columns = [human_data_source]
    dataframe.loc[:, human_data_source] = dataframe[human_data_source].str.replace('\n', '', regex=False)
    dataframe.loc[:, 'label'] = 1
    return dataframe


def generate_submission_csv_from_predictions(preds_test):
    print("Generating Submission CSV from Predictions")
    return pd.DataFrame({'id': test_essay_df["id"], 'generated': preds_test}).to_csv('submission.csv', index=False)


def create_voting_classifier():
    print("Creating Voting Classifier")
    lr_model = LogisticRegression(solver="liblinear")
    sgd_model = SGDClassifier(max_iter=1000, tol=1e-3, loss="modified_huber")
    return VotingClassifier(estimators=[('lr', lr_model), ('sgd', sgd_model)], voting='soft')


def fit_transform_dataframe(df):
    print("Fit_Transform Dataframe started.")
    continuous_text_sequence_range = (1, 3)
    transformed_df = TfidfVectorizer(ngram_range=continuous_text_sequence_range, sublinear_tf=True).fit_transform(df)
    print("Fit_Transform Dataframe completed.")
    return transformed_df


if __name__ == '__main__':
    external_df = prepare_dataframe_from_csv(EXTERNAL_AI_GENERATED_DATA_SET)
    training_essay_df = pd.read_csv(TRAIN_ESSAYS_PATH)
    training_essay_df = pd.concat([training_essay_df, external_df])
    test_essay_df = pd.read_csv("kaggle/input/llm-detect-ai-generated-text/test_essays.csv")

    data_frame = pd.concat([training_essay_df['text'], test_essay_df['text']], axis=0)

    transformed_matrix = fit_transform_dataframe(data_frame)
    num_training_rows = training_essay_df.shape[0]
    training_data = transformed_matrix[:num_training_rows]
    testing_data = transformed_matrix[num_training_rows:]

    voting_classifier = create_voting_classifier()
    voting_classifier.fit(training_data, training_essay_df.label)
    predictions = voting_classifier.predict_proba(testing_data)[:, 1]

    generate_submission_csv_from_predictions(predictions)
