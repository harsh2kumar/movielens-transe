import multiprocessing
import re
import sys
import random

def ids_only(file_path, fbs, chunkCount, chunkStart, chunkSize):
    """
    creates intermediate files containing only id's as nodes, one created for each process_id
    :param file_path: file to be read
    :param fbs: list od ids to be matched with movie_lens_freebase from KB4Rec
    :param chunkCount: current chunk being operated upon
    :param chunkStart: start byte of current chunk
    :param chunkSize: size of current chunk
    :return:
    """
    process = multiprocessing.current_process()
    # print(process.pid)
    with open(file_path, 'rb') as file:
        file.seek(chunkStart)
        lines = file.read(chunkSize).splitlines()
        with open('TempFiles/'+str(process.pid)+'.txt', 'a', encoding='utf-8') as f:
            for _ in lines:
                line=_.decode('utf-8').split('\t')
                a=re.search('m\..*[^>topic]', line[0]).group(0) if re.search('m\..*[^>topic]', line[0]) else None
                b=re.search('m\..*[^>topic]', line[2]).group(0) if re.search('m\..*[^>topic]', line[2]) else None
                if (a in fbs) or (b in fbs):
                    if a:
                        if re.match('m\..*', line[2].replace('>','').replace('<http://rdf.freebase.com/ns/', '')):
                            output=line[0].replace('>','').replace('<http://rdf.freebase.com/ns/', '')+'\t'+line[1].replace('>','').replace('<http://rdf.freebase.com/ns/', '')+'\t'+line[2].replace('>','').replace('<http://rdf.freebase.com/ns/', '')+'\n'
                            f.write(output)

def ids_topics(file_path, fbs, chunkCount, chunkStart, chunkSize):
    """
        creates intermediate files containing only id's as nodes, one created for each process_id
        :param file_path: file to be read
        :param fbs: list od ids to be matched with movie_lens_freebase from KB4Rec
        :param chunkCount: current chunk being operated upon
        :param chunkStart: start byte of current chunk
        :param chunkSize: size of current chunk
        :return:
        """
    process = multiprocessing.current_process()
    # print(process.pid)
    with open(file_path, 'rb') as file:
        file.seek(chunkStart)
        lines = file.read(chunkSize).splitlines()
        with open('TempFiles/'+str(process.pid)+'.txt', 'a', encoding='utf-8') as f:
            for _ in lines:
                line=_.decode('utf-8').split('\t')
                a=re.search('m\..*[^>Topic]', line[0]).group(0) if re.search('m\..*[^>Topic]', line[0]) else None
                b=re.search('m\..*[^>]', line[2]).group(0) if re.search('m\..*[^>]', line[2]) else None
                if (a in fbs) or (b in fbs):
                    if re.match('m\..*', line[2].replace('>','').replace('<http://rdf.freebase.com/ns/', '')):
                        output=line[0].replace('>','').replace('<http://rdf.freebase.com/ns/', '')+'\t'+line[1].replace('>','').replace('<http://rdf.freebase.com/ns/', '')+'\t'+line[2].replace('>','').replace('<http://rdf.freebase.com/ns/', '')+'\n'
                        f.write(output)

def all_rels(file_path, fbs, chunkCount, chunkStart, chunkSize):
    """
        creates intermediate files containing only id's as nodes, one created for each process_id
        :param file_path: file to be read
        :param fbs: list od ids to be matched with movie_lens_freebase from KB4Rec
        :param chunkCount: current chunk being operated upon
        :param chunkStart: start byte of current chunk
        :param chunkSize: size of current chunk
        :return:
        """
    process = multiprocessing.current_process()
    # print(process.pid)
    with open(file_path, 'rb') as file:
        file.seek(chunkStart)
        lines = file.read(chunkSize).splitlines()
        with open('TempFiles/'+str(process.pid)+'.txt', 'a', encoding='utf-8') as f:
            for _ in lines:
                line=_.decode('utf-8').split('\t')
                a=re.search('m\..*[^>]', line[0]).group(0) if re.search('m\..*[^>]', line[0]) else None
                b=re.search('m\..*[^>]', line[2]).group(0) if re.search('m\..*[^>]', line[2]) else None
                if (a in fbs) or (b in fbs):
                    output=line[0].replace('>','').replace('<http://rdf.freebase.com/ns/', '')+'\t'+line[1].replace('>','').replace('<http://rdf.freebase.com/ns/', '')+'\t'+line[2].replace('>','').replace('<http://rdf.freebase.com/ns/', '')+'\n'
                    f.write(output)

def extract_rels(file_path, fbs, chunkCount, chunkStart, chunkSize):
    """
    extracts relation fbs from current and writes them to intermediary files
    :param file_path: path of file to be read
    :param fbs: relation to be extracted
    :param chunkCount: total chunks
    :param chunkStart: start byte of current chunk
    :param chunkSize: size of current chunk
    :param indexes: a dictionary of dictionaries with keys: entity, relation
    :param indice_only: write index of line to file
    :return:
    """
    process = multiprocessing.current_process()
    # print(process.pid)
    # try:
    #     raise Exception
    # except Exception:
    #     print(random.randint(0,10))
    #     print(fbs)
    #     print(sys.getsizeof(indexes))
    #     print(id(indexes))
    #     print(len(indexes.keys()))
    with open(file_path, 'rb') as file:
        file.seek(chunkStart)
        lines = file.read(chunkSize).splitlines()
        with open('TempFiles/'+str(process.pid)+'.txt', 'a', encoding='utf-8') as f:
            for _ in lines:
                line=_.decode('utf-8').split('\t')
                if line[1].strip() == fbs:
                    f.write(_.decode('utf-8')+'\n')