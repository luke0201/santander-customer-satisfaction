# Santander Customer Satisfaction

This is my Kaggle submission code.

## Requirements

You need Python 3.6 or higher version.

The following libraries are required.

- pandas
- scikit-learn
- lightgbm

## Usage

Download the dataset using the following command.

```
kaggle competitions download -c santander-customer-satisfaction
unzip santander-customer-satisfaction.zip && rm santander-customer-satisfaction.zip
```

Then run `santander-customer-satisfaction.py` as follows.

```
python santander-customer-satisfaction.py
```

Finally, submit the result using the following command. Replace the `<submission_message>` by yours.

```
kaggle competitions submit -c santander-customer-satisfaction -f submission.csv -m <submission_message>
```
