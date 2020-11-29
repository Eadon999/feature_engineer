import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import LabelBinarizer


class DataPreprocess:
    def __init__(self, feature_model_root, config_root, config_file, is_train):
        self.feature_model_root = feature_model_root
        '''load config file'''
        self.config_df, self.config_feature_name = self.load_feature_config(
            os.path.join(config_root, config_file))
        '''load pretrain mode：oneot.pkl and so on'''
        self.feature_pretrain_model_dict = self.load_feature_pretrain_model(is_train)

    def load_feature_pretrain_model(self, is_train):
        pretrain_model_dict = {}
        for feature_name in self.config_feature_name:
            this_feature_config = self.config_df[self.config_df['combine_name'] == feature_name]
            pretrain_model_name = str(this_feature_config['pretrain_model'].values[0])
            if pretrain_model_dict.__contains__("pretrain_model_name") or pretrain_model_name in ('nan', 'unk'):
                continue
            if pretrain_model_name.endswith('.pkl') and is_train:
                pretrain_model_dict[pretrain_model_name] = self.generate_encoder_pretrain_model(this_feature_config)
            elif pretrain_model_name.endswith('.pkl') or not is_train:
                pretrain_model_dict[pretrain_model_name] = joblib.load(
                    os.path.join(self.feature_model_root, pretrain_model_name))
        return pretrain_model_dict

    def generate_encoder_pretrain_model(self, feature_config):
        type_name = feature_config['type_name'].values[0]
        pretrain_model_name = str(feature_config['pretrain_model'].values[0])
        if type_name == 'number':
            '''continue value cut to bucket'''
            bins = json.loads(feature_config['bucket_bin'].values[0])
            '''add board'''
            bins.append(np.inf)
            bins.insert(0, -np.inf)
            cut_bin_res = self.bin_bucket(bins, bins)  # output bucket label
            return self.feature_one_hot_encoder(cut_bin_res, pretrain_model_name)
        elif type_name in ['string', 'string_sparse']:
            encoder_label = json.loads(feature_config['encoder_label'].values[0])
            return self.feature_one_hot_encoder(encoder_label, pretrain_model_name)

    def read_csv(self, path, delimiter=',', use_cols=None):
        try:
            data = pd.read_csv(path, encoding='gbk', delimiter=delimiter, usecols=use_cols)
        except Exception as e:
            data = pd.read_csv(path, encoding='utf-8', delimiter=delimiter, usecols=use_cols)
        return data

    def get_col_val_delimiter(self, source_config_df):
        delimiter_dict = json.loads(source_config_df['delimiter'].values[0])
        col_sp = delimiter_dict.get('col_delimiter', '')
        value_sp = delimiter_dict.get('val_delimiter', '')
        return col_sp, value_sp

    def extract_key_value_feature(self, source, col_sp, val_sp):
        result = []
        for k_v in str(source).split(col_sp):
            k_v_sp_res = k_v.split(val_sp)
            if len(k_v_sp_res) < 2:
                continue
            key = k_v_sp_res[0]
            result.append(key)
        if len(result) < 1:
            result.append('unk')
        result = list(set(result[0:min(5, len(result))]))
        return result

    def extract_key_sparse_feature(self, source, col_sp):
        result = str(source).replace('nan', 'unk').split(col_sp)
        if len(result) < 1:
            result.append('unk')
        result = list(set(result[0:min(5, len(result))]))
        return result

    def load_feature_config(self, config_path):
        """delimiter需要用json转为dict"""
        feature_config_df = self.read_csv(config_path)
        feature_config_df = feature_config_df[feature_config_df['is_using'] == 1]
        feature_config_df['combine_name'] = feature_config_df['name']
        feature_name = list(feature_config_df['combine_name'])
        print("loaded feature_num:{},loaded feature_name:{}".format(len(feature_name), feature_name))
        return feature_config_df, feature_name

    def check_diff_feature(self, config_features, sample_features):
        config_features_len = len(config_features)
        sample_features_len = len(sample_features)
        if config_features_len < sample_features_len:
            pass
        elif config_features_len > sample_features_len:
            pass
        else:
            pass

    def feature_one_hot_encoder(self, feature_labels, model_name):
        label_binarizer = LabelBinarizer()
        one_hot_encoded = label_binarizer.fit(feature_labels)
        joblib.dump(one_hot_encoded, os.path.join(self.feature_model_root, model_name))
        return one_hot_encoded

    def bin_bucket(self, values, bins):
        bin_bucket_res = pd.cut(values, bins=bins, labels=np.arange(len(bins) - 1), right=True, include_lowest=True)
        return bin_bucket_res

    def get_feature_one_hot_value(self, feature_value, pretrain_model, is_multi_hot):
        one_hot_v = pretrain_model.transform(feature_value)
        if is_multi_hot:
            one_hot_v = np.sum(one_hot_v, axis=0)
        return one_hot_v

    def parse_original_pipeline(self, feature_values, feature_name):
        this_feature_config = self.config_df[self.config_df['combine_name'] == feature_name]
        pretrain_model_name = str(this_feature_config['pretrain_model'].values[0])
        type_name = str(this_feature_config['type_name'].values[0])
        if type_name == 'number':
            feature_values.fillna(value=0, inplace=True)
            bins = json.loads(this_feature_config['bucket_bin'].values[0])
            '''add board'''
            bins.append(np.inf)
            bins.insert(0, -np.inf)
            cut_bin_result = self.bin_bucket(feature_values, bins)
            onehot_result = self.get_feature_one_hot_value(cut_bin_result.tolist(),
                                                           self.feature_pretrain_model_dict[pretrain_model_name],
                                                           is_multi_hot=False).tolist()
            return onehot_result

        elif type_name == 'string_sparse':
            col_sp, val_sp = self.get_col_val_delimiter(this_feature_config)
            extract_key_result = [self.extract_key_sparse_feature(feature_value, col_sp) for feature_value in
                                  feature_values]
            onehot_result = [
                self.get_feature_one_hot_value(values, self.feature_pretrain_model_dict[pretrain_model_name],
                                               is_multi_hot=True).tolist() for values in extract_key_result]
            return onehot_result
        elif type_name == 'string':
            onehot_result = self.get_feature_one_hot_value(feature_values.astype(str).tolist(),
                                                           self.feature_pretrain_model_dict[pretrain_model_name],
                                                           is_multi_hot=False).tolist()
            return onehot_result

    def generate_sample_one_hot(self, sample_path_root, sample_src_fn, sample_dst_fn):

        sample_df = self.read_csv(os.path.join(sample_path_root, sample_src_fn), delimiter='\t',
                                  use_cols=self.config_feature_name)
        sample_feature = sample_df.columns
        print("sample feature_num:{},sample feature_name:{}".format(sample_feature.size, list(sample_feature)))
        self.config_feature_name.insert(0, 'label')
        print("using sample feature_num:{},using sample feature_name:{}".format(sample_feature.size,
                                                                                list(sample_feature)))
        self.config_feature_name = ['label', 'm_tag_score_s_user', 'parent_user', 'career_user']

        for feature in self.config_feature_name:
            if feature == 'label':
                continue
            feature_value_arr = sample_df[feature]
            onehot_res = self.parse_original_pipeline(feature_value_arr, feature)
            sample_df[feature] = onehot_res
        
        '''out put libsvm format file'''
        libsvm_format_result_ls = []
        for idx, item in sample_df.iterrows():
            feature_names = {}
            '''every sample'''
            concat_result = []
            for feature in self.config_feature_name:
                if feature == 'label':
                    continue
                concat_result.extend(item[feature])
            mark_one_value_idx = [str(item['label'])]
            for i in np.where(np.array(concat_result) == 1)[0]:
                mark_one_value_idx.append(str(i) + ':' + '1')
            libsvm_format_str = ' '.join(mark_one_value_idx) + '\n'
            libsvm_format_result_ls.append(libsvm_format_str)
        libsvm_sample_file = os.path.join(sample_path_root, sample_dst_fn)
        with open(libsvm_sample_file, encoding='utf-8', mode='w') as wf:
            wf.writelines(libsvm_format_result_ls)


if __name__ == '__main__':
    is_train = True
    feature_model_root = './feature_pretrain_model'
    config_root = './feature_config'
    config_file = 'feature_config.csv'
    sample_root = './sample'
    preprocessor = DataPreprocess(feature_model_root, config_root, config_file, is_train)
    preprocessor.generate_sample_one_hot(paramers)
 
