import numpy as np
from torch.utils.data import Subset


def one_hot_new(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        if each == 1:
            for n in range(3):
                df.loc[:, 'service_' + str(n)] = 0
            dicts = {'icmp': 'service_0', 'tcp': 'service_1', 'udp': 'service_2'}
            for row in df.itertuples():
                df.loc[row[0], dicts[row[2]]] = 1
            df = df.drop(each, 1)
        elif each == 2:
            for n in range(70):
                df.loc[:, 'protocol_' + str(n)] = 0
            dicts = {'IRC': 'protocol_0', 'X11': 'protocol_1', 'Z39_50': 'protocol_2', 'aol': 'protocol_3'
                , 'auth': 'protocol_4', 'bgp': 'protocol_5', 'courier': 'protocol_6', 'csnet_ns': 'protocol_7'
                , 'ctf': 'protocol_8', 'daytimne': 'protocol_9', 'discard': 'protocol_10', 'domain': 'protocol_11'
                , 'domain_u': 'protocol_12', 'echo': 'protocol_13', 'eco_i': 'protocol_14', 'ecr_i': 'protocol_15'
                , 'efs': 'protocol_16', 'exec': 'protocol_17', 'finger': 'protocol_18', 'ftp': 'protocol_19'
                , 'ftp_data': 'protocol_20', 'gopher': 'protocol_21', 'harvest': 'protocol_22',
                     'hostnames': 'protocol_23'
                , 'http': 'protocol_24', 'http_2784': 'protocol_25', 'http_443': 'protocol_26',
                     'http_8001': 'protocol_27'
                , 'imap4': 'protocol_28', 'iso_tsap': 'protocol_29', 'klogin': 'protocol_30', 'kshell': 'protocol_31'
                , 'ldap': 'protocol_32', 'link': 'protocol_33', 'login': 'protocol_34', 'mtp': 'protocol_35'
                , 'name': 'protocol_36', 'netbios_dgm': 'protocol_37', 'netbios_ns': 'protocol_38',
                     'netbios_ssn': 'protocol_39'
                , 'netstat': 'protocol_40', 'nnsp': 'protocol_41', 'nntp': 'protocol_42', 'ntp_u': 'protocol_43'
                , 'other': 'protocol_44', 'pm_dump': 'protocol_45', 'pop_2': 'protocol_46', 'pop_3': 'protocol_47'
                , 'printer': 'protocol_48', 'private': 'protocol_49', 'red_i': 'protocol_50',
                     'remote_job': 'protocol_51'
                , 'rje': 'protocol_52', 'shell': 'protocol_53', 'smtp': 'protocol_54', 'sql_net': 'protocol_55'
                , 'ssh': 'protocol_56', 'sunrpc': 'protocol_57', 'supdup': 'protocol_58', 'systat': 'protocol_59'
                , 'telnet': 'protocol_60', 'tftp_u': 'protocol_61', 'tim_i': 'protocol_62', 'time': 'protocol_63'
                , 'urh_i': 'protocol_64', 'urp_i': 'protocol_65', 'uucp': 'protocol_66', 'uucp_path': 'protocol_67'
                , 'vmnet': 'protocol_68', 'whois': 'protocol_69'}
            for row in df.itertuples():
                df.loc[row[0], dicts[row[2]]] = 1
            df = df.drop(each, 1)

        else:
            for n in range(11):
                df.loc[:, 'flag_' + str(n)] = 0
            dicts = {'OTH': 'flag_0', 'REJ': 'flag_1', 'RSTO': 'flag_2', 'RSTOS0': 'flag_3', 'RSTR': 'flag_4'
                , 'S0': 'flag_5', 'S1': 'flag_6', 'S2': 'flag_7', 'S3': 'flag_8', 'SF': 'flag_9', 'SH': 'flag_10'
                , 'SHR': 'flag_10', 'RSTRH': 'flag_10'}
            for row in df.itertuples():
                df.loc[row[0], dicts[row[2]]] = 1
            df = df.drop(each, 1)

    return df



def normalize_new(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with normalized specified features
    """
    result = df.copy()  # do not touch the original df
    featuremax = [57715, 1379963888, 1309937401, 1, 3, 3, 101, 5, 1, 7479, 1, 2, 7468, 100, 5, 9, 0, 1, 1, 511,
                  511, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 255, 255, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1]
    featuremin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0]
    for i, feature_name in enumerate(cols):
        max_value = featuremax[i]
        min_value = featuremin[i]
        if max_value > min_value:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result



class CustomSubset_two(Subset):
    '''A custom subset class with customizable data transformation'''

    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]

        x = np.reshape(x, (1, x.shape[0]))

        return x, y
