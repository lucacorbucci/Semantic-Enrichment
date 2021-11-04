import pandas as pd


class Utils:
    def find_diabetes_HADMID(self, df: pd.DataFrame) -> list[str]:
        HADM_ID = set()
        for _, row in df.iterrows():
            if "25000" in eval(row["ICD9_CODE"]):
                HADM_ID.add(row["HADM_ID"])
        return list(HADM_ID)

    def extract_diabete_only(self, df: pd.DataFrame) -> pd.DataFrame:
        HADM_ID = self.find_diabetes_HADMID(df)
        reduced_df = df[df["HADM_ID"].isin(HADM_ID)]
        return reduced_df

    def remove_abbreviations(self, df, abbreviations):
        extended_text = []
        extended_tokens = []
        for _, row in df.iterrows():
            text = row["TEXT"]
            token = eval(row["Token"])
            new_token = []
            new_text = []
            for word in token:
                if word in abbreviations:
                    new_token.append(abbreviations.get(word))
                else:
                    new_token.append(word)
            extended_tokens.append(new_token)
            for word in text.split():
                if word in abbreviations:
                    new_text.append(abbreviations.get(word))
                else:
                    new_text.append(word)
            tmp = " ".join(str(v) for v in new_text)
            extended_text.append(tmp)
        return extended_text, extended_tokens