'''
This is the main function of the PE classification of this program with streamlit
'''

from keras import backend as K
import pefile
import os
import dill as pickle
import sys
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import keras


def get_entropy(data: list):
    # For calculating the entropy
    if len(data) == 0:
        return 0.0
    # Compute the probability distribution of the data
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    # Calculate the entropy
    entropy = stats.entropy(probabilities, base=2)
    return entropy

# For extracting the resources part


def get_resources(pe: object) -> list:
    """Extract resources :
    [entropy, size]"""
    resources = []
    if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
        try:
            for resource_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
                if hasattr(resource_type, 'directory'):
                    for resource_id in resource_type.directory.entries:
                        if hasattr(resource_id, 'directory'):
                            for resource_lang in resource_id.directory.entries:
                                data = pe.get_data(
                                    resource_lang.data.struct.OffsetToData, resource_lang.data.struct.Size)
                                size = resource_lang.data.struct.Size
                                entropy = get_entropy(data)
                                resources.append([entropy, size])
        except Exception as e:
            return resources
    return resources


def get_version_info(pe: object) -> dict:
    """Return version infos"""
    res = {}
    for fileinfo in pe.FileInfo:
        if fileinfo.Key == 'StringFileInfo':
            for st in fileinfo.StringTable:
                for entry in st.entries.items():
                    res[entry[0]] = entry[1]
        if fileinfo.Key == 'VarFileInfo':
            for var in fileinfo.Var:
                res[var.entry.items()[0][0]] = var.entry.items()[0][1]
    if hasattr(pe, 'VS_FIXEDFILEINFO'):
        res['flags'] = pe.VS_FIXEDFILEINFO.FileFlags
        res['os'] = pe.VS_FIXEDFILEINFO.FileOS
        res['type'] = pe.VS_FIXEDFILEINFO.FileType
        res['file_version'] = pe.VS_FIXEDFILEINFO.FileVersionLS
        res['product_version'] = pe.VS_FIXEDFILEINFO.ProductVersionLS
        res['signature'] = pe.VS_FIXEDFILEINFO.Signature
        res['struct_version'] = pe.VS_FIXEDFILEINFO.StrucVersion
    return res


def extract_infos(fpath: str) -> dict:
    # extract the info for a given file using pefile
    fpath = os.getcwd() + f'\\{fpath}'
    res = {}
    pe = pefile.PE(fpath)
    res['Machine'] = pe.FILE_HEADER.Machine
    res['SizeOfOptionalHeader'] = pe.FILE_HEADER.SizeOfOptionalHeader
    res['Characteristics'] = pe.FILE_HEADER.Characteristics
    res['MajorLinkerVersion'] = pe.OPTIONAL_HEADER.MajorLinkerVersion
    res['MinorLinkerVersion'] = pe.OPTIONAL_HEADER.MinorLinkerVersion
    res['SizeOfCode'] = pe.OPTIONAL_HEADER.SizeOfCode
    res['SizeOfInitializedData'] = pe.OPTIONAL_HEADER.SizeOfInitializedData
    res['SizeOfUninitializedData'] = pe.OPTIONAL_HEADER.SizeOfUninitializedData
    res['AddressOfEntryPoint'] = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    res['BaseOfCode'] = pe.OPTIONAL_HEADER.BaseOfCode
    try:
        res['BaseOfData'] = pe.OPTIONAL_HEADER.BaseOfData
    except AttributeError:
        res['BaseOfData'] = 0
    res['ImageBase'] = pe.OPTIONAL_HEADER.ImageBase
    res['SectionAlignment'] = pe.OPTIONAL_HEADER.SectionAlignment
    res['FileAlignment'] = pe.OPTIONAL_HEADER.FileAlignment
    res['MajorOperatingSystemVersion'] = pe.OPTIONAL_HEADER.MajorOperatingSystemVersion
    res['MinorOperatingSystemVersion'] = pe.OPTIONAL_HEADER.MinorOperatingSystemVersion
    res['MajorImageVersion'] = pe.OPTIONAL_HEADER.MajorImageVersion
    res['MinorImageVersion'] = pe.OPTIONAL_HEADER.MinorImageVersion
    res['MajorSubsystemVersion'] = pe.OPTIONAL_HEADER.MajorSubsystemVersion
    res['MinorSubsystemVersion'] = pe.OPTIONAL_HEADER.MinorSubsystemVersion
    res['SizeOfImage'] = pe.OPTIONAL_HEADER.SizeOfImage
    res['SizeOfHeaders'] = pe.OPTIONAL_HEADER.SizeOfHeaders
    res['CheckSum'] = pe.OPTIONAL_HEADER.CheckSum
    res['Subsystem'] = pe.OPTIONAL_HEADER.Subsystem
    res['DllCharacteristics'] = pe.OPTIONAL_HEADER.DllCharacteristics
    res['SizeOfStackReserve'] = pe.OPTIONAL_HEADER.SizeOfStackReserve
    res['SizeOfStackCommit'] = pe.OPTIONAL_HEADER.SizeOfStackCommit
    res['SizeOfHeapReserve'] = pe.OPTIONAL_HEADER.SizeOfHeapReserve
    res['SizeOfHeapCommit'] = pe.OPTIONAL_HEADER.SizeOfHeapCommit
    res['LoaderFlags'] = pe.OPTIONAL_HEADER.LoaderFlags
    res['NumberOfRvaAndSizes'] = pe.OPTIONAL_HEADER.NumberOfRvaAndSizes

    # Sections
    res['SectionsNb'] = len(pe.sections)
    entropy = list(map(lambda x: x.get_entropy(), pe.sections))
    res['SectionsMeanEntropy'] = sum(entropy)/float(len((entropy)))
    res['SectionsMinEntropy'] = min(entropy)
    res['SectionsMaxEntropy'] = max(entropy)
    raw_sizes = list(map(lambda x: x.SizeOfRawData, pe.sections))
    res['SectionsMeanRawsize'] = sum(raw_sizes)/float(len((raw_sizes)))
    res['SectionsMinRawsize'] = min(raw_sizes)
    res['SectionsMaxRawsize'] = max(raw_sizes)
    virtual_sizes = list(map(lambda x: x.Misc_VirtualSize, pe.sections))
    res['SectionsMeanVirtualsize'] = sum(
        virtual_sizes)/float(len(virtual_sizes))
    res['SectionsMinVirtualsize'] = min(virtual_sizes)
    res['SectionMaxVirtualsize'] = max(virtual_sizes)

    # Imports
    try:
        res['ImportsNbDLL'] = len(pe.DIRECTORY_ENTRY_IMPORT)
        imports = sum([x.imports for x in pe.DIRECTORY_ENTRY_IMPORT], [])
        res['ImportsNb'] = len(imports)
        res['ImportsNbOrdinal'] = 0
    except AttributeError:
        res['ImportsNbDLL'] = 0
        res['ImportsNb'] = 0
        res['ImportsNbOrdinal'] = 0

    # Exports
    try:
        res['ExportNb'] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
    except AttributeError:
        # No export
        res['ExportNb'] = 0
    # Resources
    resources = get_resources(pe)
    res['ResourcesNb'] = len(resources)
    if len(resources) > 0:
        entropy = list(map(lambda x: x[0], resources))
        res['ResourcesMeanEntropy'] = sum(entropy)/float(len(entropy))
        res['ResourcesMinEntropy'] = min(entropy)
        res['ResourcesMaxEntropy'] = max(entropy)
        sizes = list(map(lambda x: x[1], resources))
        res['ResourcesMeanSize'] = sum(sizes)/float(len(sizes))
        res['ResourcesMinSize'] = min(sizes)
        res['ResourcesMaxSize'] = max(sizes)
    else:
        res['ResourcesNb'] = 0
        res['ResourcesMeanEntropy'] = 0
        res['ResourcesMinEntropy'] = 0
        res['ResourcesMaxEntropy'] = 0
        res['ResourcesMeanSize'] = 0
        res['ResourcesMinSize'] = 0
        res['ResourcesMaxSize'] = 0

    # Load configuration size
    try:
        res['LoadConfigurationSize'] = pe.DIRECTORY_ENTRY_LOAD_CONFIG.struct.Size
    except AttributeError:
        res['LoadConfigurationSize'] = 0

    # Version configuration size
    try:
        version_infos = get_version_info(pe)
        res['VersionInformationSize'] = len(version_infos.keys())
    except AttributeError:
        res['VersionInformationSize'] = 0
    return res


def covert_to_png(fpath: str, img_size: tuple = (64, 64, 3)) -> Image:
    # This function allows us to process our files into png images##
    with open(fpath, 'rb') as file:
        binary_data = file.read()
    file.close()

    # Convert the bytes to a numpy array
    file_array = np.frombuffer(binary_data, dtype=np.uint8)

    # Resize the array to (64, 64)
    resized_array = np.resize(file_array, img_size)

    # Create a grayscale PIL Image from the resized array
    image = Image.fromarray(resized_array, mode='RGB')

    return image


# Loading the malware detector and features
with open('Classifier\PE\pickel_malware_detector.pkl', 'rb') as file:
    detector = pickle.load(file)
file.close()
with open('Classifier\PE\\features.pkl', 'rb') as file2:
    features = pickle.load(file2)
file2.close()

# Loading the malware classifier and list of malware family classes


def recall_m(y_test, y_pred):
    true_positives = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_test, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_test, y_pred):
    true_positives = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_test, y_pred):
    precision = precision_m(y_test, y_pred)
    recall = recall_m(y_test, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


classifier = tf.keras.saving.load_model(
    filepath='Classifier\Malware_Classifier\\tf_save_model.keras',
    custom_objects={
        'f1_m': f1_m,
        'precision_m': precision_m,
        'recall_m': recall_m
    }
)
with open('Classifier\Malware_Classifier\Malware_classes.pkl', 'rb') as file4:
    class_names = pickle.load(file4)
file4.close()


class PE_scanner:
    '''
    A class to represent an PE scanner 
    ...
    Attributes
    ----------
    clf: RandomForestClassifier model
        a trained RandomForestClassifier model that uses features to identify a malware

    features: list
        a list of neccessary features in PE Header

    extract_infos: function
        a function to get info from PE file
    '''

    def __init__(self):
        self.detector: RandomForestClassifier = detector
        self.features: list = features
        self.extract_infos: function = extract_infos
        self.classifier: keras.Sequential = classifier
        self.malware_classes: list = class_names
        self.covert_to_png = covert_to_png

    def PE_scan(self, fpath: str) -> int:
        '''
        Return a result if the give PE file is malicious or benign

        Parameters
        ----------
        fpath: file path

        Returns
        -------
        0: malicious
        1: legitmate
        '''
        try:
            data = self.extract_infos(fpath)
        except:
            data = {}

        if data != {}:
            pe_features = list(map(lambda x: data[x], features))
            pe_result = self.detector.predict([pe_features])[0]
        else:
            pe_result = 1
        return pe_result

    def PE_mal_classify(self, fpath: str) -> str:
        png = self.covert_to_png(fpath)
        img_array = keras.utils.img_to_array(png)
        img_array = tf.expand_dims(img_array, 0)
        predictions = self.classifier.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        return class_names[np.argmax(score)]
