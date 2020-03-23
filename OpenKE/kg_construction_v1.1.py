# before running this code, set Working Directory to Project Directory in Run->Edit Configurations->Working Directory

import operator
import csv
import pickle
import shutil
import math, random
from datetime import datetime
import multiprocessing as mp, os
import util_v2
import pandas as pd
import random
import numpy as np
import contextlib
from easydict import EasyDict as edict
import sys
import copy
import pprint


class MovieLens():
  file_path = ''
  one_step_graph = ''
  data_freebase = ''
  relations = {}
  fbs = []
  cores = 0
  df = ''
  mid_fid = ''
  batch_size = 0
  entity_indice_dict, relation_indice_dict = {}, {}
  indices = {}

  def __init__(self):
    # init
    self.cores = os.cpu_count()
    self.one_step_graph = r"data/1step_graph/graph_movie_1step.txt"
    self.file_size = os.path.getsize(self.one_step_graph)
    self.data_freebase = r"data_processed/data_freebase.txt"
    self.ratings = r"data/ml-20m/ratings.csv"
    self.test_data = r'data_processed/test/test_data.txt'

    # create list of all freebase movie ids
    with open(r"data/KB4Rec/ml2fb.txt", 'r', encoding='utf-8') as f:
      self.fbs = [line.split()[1] for line in f]

    # create map from movie_id to freebase_id
    self.df = pd.read_csv(r"data/KB4Rec/ml2fb.txt", sep='\t', header=None)
    self.mid_fid = dict(zip(self.df[0], self.df[1]))

    # create directories
    os.makedirs(r"data_processed/relations", exist_ok=True)
    os.makedirs(r"data_processed/relations_indices", exist_ok=True)
    os.makedirs(r"data_processed/entities", exist_ok=True)
    os.makedirs(r"data_processed/new_entities", exist_ok=True)
    os.makedirs(r"data_processed/test", exist_ok=True)
    os.makedirs(r"data_processed/test_indices", exist_ok=True)
    os.makedirs(r"TempFiles", exist_ok=True)

    # init relations
    self.relations_ordered_used = r'data_processed/relations_ordered_used.txt'
    self.entities_dir = r'data_processed/entities'
    self.relations_dir = r'data_processed/relations'
    self.test_dir = r'data_processed/test'
    self.relations_indices_dir = r'data_processed/relations_indices'
    self.test_indices_dir = r'data_processed/test_indices'
    self.relations_ordered = r"data_processed/relations_ordered.txt"

    #new dirs
    self.new_relations_indices_dir = r'data_processed/new_relations_indices'
    self.new_test_indices_dir = r'data_processed/new_test_indices'

    self.batch_size = 64

  def silent_remove(self, filepath):
    """
removes file if it exists
:param filename: path of file to be removed
:return:
"""
    with contextlib.suppress(FileNotFoundError):
      os.remove(filepath)

  def flush_temp_files(self):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(r'TempFiles'):
      for file in f:
        if '.txt' in file:
          files.append(os.path.join(r, file))

    for f in files:
      self.silent_remove(f)

  def chunkify(self, fname, size=1024 * 1024):
    """
    returns chunked file
    :param fname: name of file to be chunked
    :param size: size of chunks
    """
    chunkCount = 0
    totalChunks = self.file_size / size
    print('Total chunks', math.ceil(totalChunks))
    fileEnd = os.path.getsize(fname)
    with open(fname, 'rb') as f:
      chunkEnd = f.tell()
      while True:
        chunkStart = chunkEnd
        f.seek(size, 1)
        f.readline()
        chunkEnd = f.tell()
        chunkCount += 1
        yield chunkCount, chunkStart, chunkEnd - chunkStart
        if chunkEnd > fileEnd:
          break

  def process_files(self, file_path, path, process_wrapper, task):
    """

    :param file_path: source file to be processed
    :param path: storage location of file
    :param process_wrapper: function to be called to process file
    :param task: descrbe task being executed
    """
    s = datetime.now()

    # init objects
    pool = mp.Pool(self.cores)
    jobs = []
    self.file_path = file_path

    # create jobs
    print('Task: ', task)
    for chunkCount, chunkStart, chunkSize in self.chunkify(self.file_path):
      jobs.append(
        pool.apply_async(process_wrapper, (self.file_path, self.fbs, chunkCount, chunkStart, chunkSize)))

    # wait for all jobs to finish
    for job in jobs:
      job.get()

    print('Processing completed. Time elapsed: ', datetime.now() - s)

    # combine files
    print('Combining files')

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(r'TempFiles'):
      for file in f:
        if '.txt' in file:
          files.append(os.path.join(r, file))

    with open(path, 'w', encoding='utf-8') as op:
      for f in files:
        with open(f, 'r', encoding='utf-8') as cur:
          for line in cur:
            op.write(line)
        self.silent_remove(f)

    # clean up
    pool.close()
    e = datetime.now()
    print('Total time elapsed: ', e - s)

  def make_datasets(self):
    """
driver function to create initial files for custom freebase_database.txt
:return:
"""
    self.flush_temp_files()
    self.process_files(self.one_step_graph, r"data_processed/data_ids_only.txt", util_v2.ids_only, 'IDs only')

  # self.process_files(self.one_step_graph, r"data_processed/data_ids_topics.txt", util_v2.ids_topics, 'IDs n Topics')
  # self.process_files(self.one_step_graph, r"data_processed/data_all_rels.txt", util_v2.all_rels, 'All relations')

  def freebase_dataset(self):
    """
    append userid user.watched.movie movieid to data_freebase.txt
    """
    print('Appending user.watched.movie relation to freebase dataset')
    s = datetime.now()
    self.flush_temp_files()
    self.silent_remove(self.data_freebase)
    shutil.copy(r"data_processed/data_ids_only.txt", self.data_freebase)
    user_w_mov = {}
    f2su = open(r"data_processed/skipped_users.txt", 'w')
    with open(self.ratings, 'r') as csvfile:
      csvreader = csv.reader(csvfile, delimiter=',')
      # This skips the first row of CSV file.
      # csvreader.next() also works in Python 2.
      next(csvreader)
      with open(self.test_data, 'w') as f2t:
        with open(self.data_freebase, 'a') as f:
          for row in csvreader:
            if int(row[1]) in self.mid_fid.keys():
              if row[0] in user_w_mov:
                user_w_mov[row[0]].append(row[1])
              else:
                user_w_mov[row[0]] = []
          for key, values in user_w_mov.items():
            if int(len(values)*0.7)<5:
              f2su.write(str(key) + '\t' + str(len(values)) + '\t' + str(int(len(values)*0.7)) + '\n')
              print('Skipping user: ', key)
              continue
            # shuffle watched movies array in-place
            np.random.shuffle(values)
            train_idx = int(len(values) * 0.7)
            # 70%-30% train-test split
            for movie in values[:train_idx]:
              f.write(str(key) + '\t' + 'user_watched.user_id.movie' + '\t' + 'movie_id_'+str(movie) + '\n')
            for movie in values[train_idx:]:
              f2t.write(str(key) + '\t' + 'user_watched.user_id.movie' + '\t' + 'movie_id_'+str(movie) + '\n')
    print('Completed. Time elapsed: ', datetime.now() - s)

  def get_relations(self):
    """
find count of all relations
store relations where frequency>10000
"""
    self.relations = {}

    with open(self.data_freebase, 'r', encoding='utf-8') as f:
      for line in f:
        # print(line)
        self.relations[line.split()[1]] = self.relations.get(line.split()[1], 0) + 1

    sorted_relations = sorted(self.relations.items(), key=operator.itemgetter(1), reverse=True)

    with open(self.relations_ordered, 'w', encoding='utf-8') as f:
      for ele in sorted_relations:
        f.write(ele[0] + '\t' + str(ele[1]) + '\n')

    with open(self.relations_ordered_used, 'w', encoding='utf-8') as f:
      for ele in sorted_relations:
        if int(ele[1]) > 10000:
          f.write(ele[0].strip() + '\n')

  def make_relations(self):
    """
create a new file for each qualified relation and store all possible interactions
:param indexes_only: when True, we write only indexs of entities and relations to file
:return:
"""
    self.flush_temp_files()
    with open(self.relations_ordered_used, 'r', encoding='utf-8') as f:
      for ele in f:
        self.fbs = ele.strip()
        cur_path = os.path.join(r"data_processed/relations", ele.strip() + '.txt')
        self.process_files(self.data_freebase, cur_path, util_v2.extract_rels,
                           'Relations extraction: ' + ele.strip())
        self.make_indiced_relations()

  def make_indiced_relations(self, new_entities_dir=None):
    """
    creates relations_indices with newly created entity types from entity permutaions in MovieLensDataLoader
    :return:
    """
    if not new_entities_dir:
      # recognize entities
      self.make_entities()
      # relations with indices
      print('\nCreating relations with indices\n')
      s0 = datetime.now()
      self.get_indices()
      dirs_to_explore = {self.relations_dir: self.relations_indices_dir, self.test_dir: self.test_indices_dir}
      for dirs, indice_dirs in dirs_to_explore.items():
        for r, d, f in os.walk(dirs):
          for file in f:
            if '.txt' in file:
              s = datetime.now()
              print(file)
              with open(os.path.join(r, file), 'r') as f2r:
                with open(os.path.join(indice_dirs, file), 'w') as f2w:
                  for lines in f2r:
                    line = lines.split()
                    head_entity = line[1].strip().split('.')[1]
                    tail_entity = line[1].strip().split('.')[2]
                    output = str(self.indices['entity'][head_entity][line[0].strip()]) + '\t' + str(
                      self.indices['relation'][line[1].strip()]) + '\t' + \
                             str(self.indices['entity'][tail_entity][line[2].strip()]) + '\n'
                    f2w.write(output)
              print('Processing completed. Time elapsed: ', datetime.now() - s)
      print('Total time elapsed: ', datetime.now() - s0)
    else:
      print('\nRe-creating relations with indices\n')
      s0 = datetime.now()
      self.get_indices(new_entities_dir)
      dirs_to_explore = {self.relations_dir: self.relations_indices_dir, self.test_dir: self.test_indices_dir}
      for dirs, indice_dirs in dirs_to_explore.items():
        for r, d, f in os.walk(dirs):
          for file in f:
            if '.txt' in file:
              s = datetime.now()
              print(file)
              with open(os.path.join(r, file), 'r') as f2r:
                with open(os.path.join(indice_dirs, file), 'w') as f2w:
                  for lines in f2r:
                    line = lines.split()
                    head_entity = line[1].strip().split('.')[1]
                    tail_entity = line[1].strip().split('.')[2]
                    output = str(self.indices['composite_entities'][head_entity][line[0].strip()]) + '\t' + str(
                      self.indices['relation'][line[1].strip()]) + '\t' + \
                             str(self.indices['entity'][tail_entity][line[2].strip()]) + '\n'
                    f2w.write(output)
              print('Processing completed. Time elapsed: ', datetime.now() - s)
      print('Total time elapsed: ', datetime.now() - s0)

  def make_entities(self):
    """
creates entities from relation files
:return: None
"""
    entities = {}
    # r=root, d=directories, f = files
    for r, d, f in os.walk(self.relations_dir):
      for file in f:
        if '.txt' in file:
          ele = file.split(sep='.')
          entities[ele[1]] = entities.get(ele[1], set())
          entities[ele[2]] = entities.get(ele[2], set())

          with open(os.path.join(r, file), 'r') as f2r:
            for line in f2r:
              entity_ele = line.split()
              entities[ele[1]].add(entity_ele[0])
              entities[ele[2]].add(entity_ele[2])
    # add entities from test_set
    for r, d, f in os.walk(self.test_dir):
      for file in f:
        with open(os.path.join(r, file), 'r') as f2r:
          for line in f2r:
            entity_ele = line.split()
            et_head = entity_ele[1].split('.')[1]
            et_tail = entity_ele[1].split('.')[2]
            entities[et_head].add(entity_ele[0])
            entities[et_tail].add(entity_ele[2])

    for key, values in entities.items():
      with open(os.path.join(self.entities_dir, key + '.txt'), 'w') as f2w:
        for v in values:
          f2w.write(v + '\n')
      print('Completed entity: ', key)

  def get_indices(self, new_entities_dir=None):
    """
populates the indexes of relations and entities in their respective dictionaries
:return: None
"""
    if new_entities_dir:
      for r, d, f in os.walk(new_entities_dir):
        self.indice_dict['composite_entities'] = {}
        for file in f:
          if '.txt' in file:
            with open(os.path.join(r, file), 'r') as cur:
              count = 0
              for line in cur:
                self.indice_dict['composite_entities'][line.strip()] = self.indice_dict[
                  'composite_entities'].get(line.strip(), count)
                count += 1
    else:
      for r, d, f in os.walk(self.entities_dir):
        for file in f:
          if '.txt' in file:
            entity_name = file.replace('.txt', '').strip()
            self.entity_indice_dict[entity_name] = {}
            with open(os.path.join(r, file), 'r') as cur:
              count = 0
              for line in cur:
                self.entity_indice_dict[entity_name][line.strip()] = self.entity_indice_dict[
                  entity_name].get(line.strip(), count)
                count += 1

    with open(self.relations_ordered_used, 'r') as file:
      count = 0
      for line in file:
        relation_name = line.strip()
        self.relation_indice_dict[relation_name] = self.relation_indice_dict.get(relation_name, count)
        count += 1

    self.indices['entity'] = self.entity_indice_dict
    self.indices['relation'] = self.relation_indice_dict


class MovieLensDataLoader():
  def __init__(self, batch_size=64):
    # init relations
    cwd = os.getcwd()
    self.relations_ordered_used = os.path.join(cwd, r'data_processed/relations_ordered_used.txt')
    self.relations_ordered = os.path.join(cwd, r"data_processed/relations_ordered.txt")
    self.entities_dir = os.path.join(cwd, r'data_processed/entities')
    self.new_entities_dir = os.path.join(cwd, r'data_processed/new_entities')
    self.relations_dir = os.path.join(cwd, r'data_processed/relations')
    self.edict_relations_pkl = os.path.join(cwd, r'data_processed/edict_relations.pkl')
    self.edict_relations_dict_pkl = os.path.join(cwd, r'data_processed/edict_relations_dict.pkl')
    self.relations_indices_dir = os.path.join(cwd, r'data_processed/relations_indices')
    self.new_relations_indices_dir = os.path.join(cwd, r'data_processed/new_relations_indices')
    self.batch_size = batch_size
    # change unacceptable entity and relation names
    self.lexicon = {'type': 'types'}
    self.finished_word_num = 0
    self.finished_word_num_lr = 0
    self._has_next = True
    self.total_samples = 0
    self.ez_dataset_relations = []
    self.word_size = 0
    self.number_of_relations = 0

    os.makedirs(r"data_processed/new_entities", exist_ok=True)

  def reset(self):
    # resets finished_word_counter and has_next to True
    self.finished_word_num = 0
    self._has_next = True

  def get_edict(self):
    """

    :return: return edict of all entities and relations
    """
    word_size = 0
    number_of_relations = 0
    ez_dataset = edict(entities=edict(), relations=edict(), entity_instance=edict(), word_size=0)
    print('Parsing entities')
    # r=root, d=directories, f = files
    for r, d, f in os.walk(self.entities_dir):
      for file in f:
        if '.txt' in file:
          ele = file.strip().split(sep='.')
          ez_dataset.entities[self.lexicon.get(ele[0], ele[0])] = edict(vocab_size=0)
          with open(os.path.join(r, file), 'r') as f2r:
            count = 0
            for line in f2r:
              entity_ele = line.split()[0]
              ez_dataset.entities[entity_ele] = ez_dataset.entities.get(entity_ele, {})
              ez_dataset.entities[entity_ele].ent_cnt=ez_dataset.entities[entity_ele].get('ent_cnt', 0)+1
              ez_dataset.entities[entity_ele].ents = (ez_dataset.entities[entity_ele].get('ents', []))+[str(ele[0])]
              # ez_entities[ele[0]].data.append(entity_ele[0])
              count += 1
            ez_dataset.entities[self.lexicon.get(ele[0], ele[0])].vocab_size = count

    # stats on entity_permutations
    count_permute = edict()  # Stores count of permutations
    entity_unique_permute = edict()  # stores entity_instance->entity_type_permutation
    with open(os.path.join(os.getcwd(), r'data_processed/entity_tomy.txt'), 'w') as f:
      for k, v in ez_dataset.entities.items():
        # if 'ent_cnt' in v and v.ent_cnt>1:
        if 'ent_cnt' in v:
          # print(k, v)
          f.write(str(k)+' '+str(v)+'\n')
          count_permute[str(sorted(v.ents))] = count_permute.get(str(sorted(v.ents)), 0) + 1
          entity_unique_permute[k]=entity_unique_permute.get(k, str(sorted(v.ents)))
    with open(os.path.join(os.getcwd(), r'data_processed/entity_tomy_perm.txt'), 'w') as f2:
      pprint.pprint(count_permute, f2)
    # return
    # make new relations_indices based on a new directory
    inv_map={}
    for k, v in entity_unique_permute.items():
      print(('-'*10))
      inv_map[v]=inv_map.get(v, [])+[k]
    # inv_map={}
    # inv_map = {v: inv_map.get(k, [])+[k] for k, v in entity_unique_permute.items()}
    # create new entity types
    counter=0
    for k, v in inv_map.items():
      counter+=1
      with open(os.path.join(self.new_entities_dir, 'new_entity_'+str(counter)+'.txt'), 'w') as f:
        for entities in v:
          f.write(str(entities)+'\n')
    # create new indiced relations
    MovieLens().make_indiced_relations(self.new_entities_dir)
    print('Parsing relations')
    for r, d, f in os.walk(self.relations_indices_dir):
      for file in f:
        print(file)
        if '.txt' in file:
          ele = file.strip()
          number_of_relations += 1
          ez_dataset.relations[self.lexicon.get(ele, ele)] = edict(vocab_size=0)
          with open(os.path.join(r, file), 'r') as f2r:
            count = 0
            for line in f2r:
              # old way to get entity types
              # et_head = self.lexicon.get(ele.split('.')[0], ele.split('.')[0])
              # et_tail = self.lexicon.get(ele.split('.')[2], ele.split('.')[2])

              # new way to get entity types
              entity_ele = line.strip().split()
              et_head = entity_unique_permute[entity_ele[0]]
              et_tail = entity_unique_permute[entity_ele[2]]

              ez_dataset.entities[et_head] = ez_dataset.entities.get(et_head, {})
              ez_dataset.entities[et_tail] = ez_dataset.entities.get(et_tail, {})
              ez_dataset.entities[et_head][entity_ele[0]] = ez_dataset.entities[
                                                              et_head].get(
                entity_ele[0], 0) + 1
              ez_dataset.entities[et_tail][entity_ele[2]] = ez_dataset.entities[
                                                              et_tail].get(
                entity_ele[2], 0) + 1
              # ez_dataset.relation[ele[0]].data.append(entity_ele[0])
              # if et_head not in ez_dataset.entities.entity_instance.get(entity_ele[0], []):
              #   ez_dataset.entities.entity_instance[entity_ele[0]] = ez_dataset.entities.entity_instance.get(entity_ele[0], [])+[str(et_head)]
              # if et_tail not in ez_dataset.entities.entity_instance.get(entity_ele[2], []):
              #   ez_dataset.entities.entity_instance[entity_ele[2]] = ez_dataset.entities.entity_instance.get(entity_ele[2], [])+[str(et_tail)]

              count += 1
              word_size += 1
            ez_dataset.relations[self.lexicon.get(ele, ele)].vocab_size = count
    # break
    ez_dataset.word_size = word_size
    ez_dataset.number_of_relations = number_of_relations
    with open(self.edict_relations_pkl, 'wb') as pf:
      pickle.dump(ez_dataset, pf)

    # check for duplicated instances of entity_instance in different entity_types
    # with open(os.path.join(os.getcwd(), r'data_processed/entity_err_instances.txt'), 'w') as f:
    #   for k, v in ez_dataset.entity_instance.items():
    #     if len(v)>1:
    #       f.write(str(k) + ' ' + str(v) + '\n')

  def load_edict(self):
    # with open(self.edict_relations_pkl, 'rb') as pf:
    ez_dataset = pickle.load(open(self.edict_relations_pkl, 'rb'))
    self.ez_dataset_relations = pickle.load(open(self.edict_relations_dict_pkl, 'rb'))
    self.total_samples = int(ez_dataset.word_size / (ez_dataset.number_of_relations * self.batch_size))
    return ez_dataset

  # def random_sampler(self, filename, k):
  # 	"""
  #     :param k: batch_size
  #     :return: k random lines from a file- user_defined_relation.txt
  #     """
  # 	sample = []
  # 	with open(filename, 'rb') as f:
  # 		f.seek(0, 2)
  # 		filesize = f.tell()
  #
  # 		random_set = sorted(random.sample(range(filesize), k))
  #
  # 		for i in range(k):
  # 			f.seek(random_set[i])
  # 			# Skip current line (because we might be in the middle of a line)
  # 			f.readline()
  # 			if not f.readline(): i-=1
  # 			# Append the next line to the sample set
  # 			sample.append(list(map(np.int, f.readline().decode('utf-8').rstrip().split())))
  #
  # 	return np.array(sample)

  def random_sampler(self, relations_dict, relation, k):
    random_set = sorted(random.sample(range(len(relations_dict[relation])), k))
    sample = []
    for i in range(k):
      sample.append(relations_dict[relation][random_set[i]])
    return np.array(sample)

  def build_edict_random_sampler(self):
    with open(self.relations_ordered_used, 'r') as file:
      temp = edict()
      for ele in file:
        with open(os.path.join(self.relations_indices_dir, ele.strip() + '.txt'), 'r') as f2r:
          relation = ele.replace('.txt', '').replace('.', '_').strip()
          all_lines = []

          lines = f2r.readlines()[:-1]
          for line in lines:
            all_lines.append(list(map(np.int, line.rstrip().split())))
          temp[relation] = all_lines

    with open(self.edict_relations_dict_pkl, 'wb') as f2w:
      pickle.dump(temp, f2w)

  def get_batch(self):
    """
    creates a batch sampling _size_ facts for each relation
    :param size: frequency of facts for each relation
    :return: return a matrix [[batch_size facts] x 63] for all relations inclusive
    """
    batch = []
    self.finished_word_num += 1
    self.finished_word_num_lr += 1
    with open(self.relations_ordered_used, 'r') as file:
      for ele in file:
        file_path = os.path.join(self.relations_indices_dir, ele.strip() + '.txt')
        line = ele.strip().split('.')
        temp = edict()
        temp.head_entity = self.lexicon.get(line[0], line[0])
        temp.tail_entity = self.lexicon.get(line[2], line[2])
        # temp.relation = self.lexicon.get(line[1], line[1])
        temp.relation = ele.replace('.txt', '').replace('.', '_').strip()
        temp.sample = np.array(self.random_sampler(self.ez_dataset_relations, temp.relation, self.batch_size))
        batch.append(temp)
    if self.finished_word_num > self.total_samples:
      self._has_next = False
    return batch

  def has_next(self):
    return self._has_next


def main():
  # processing completed in approximately 4:30 minutes(270 s) for 16 cores Intel Xeon Gold 6130 @ 2.10GHz
  s=datetime.now()
  # test if CWD is correct
  print(os.getcwd())
  sol = MovieLens()
  # # limit max usage to 25%
  # # comment below line for faster procesing
  # # sol.cores=int(sol.cores/4)
  print('Used CPU cores: ', sol.cores)

  # print('Construct freebase dataset')
  # # # create initial files
  # sol.make_datasets()
  # #
  # # # create freebase dataset
  # sol.freebase_dataset()
  #
  # print('Construct freebase relations')
  # # create relations
  # sol.get_relations()
  # sol.make_relations()
  # print('Completed execution in: ', datetime.now()-s)
  #
  # # Run MovieLensDataLoader to create essential pickle files
  print('-'*50+' Creating pickle files')
  sol_Loader=MovieLensDataLoader(batch_size=64)
  sol_Loader.get_edict()
  # sol_Loader.build_edict_random_sampler()
  # dataset = sol_Loader.load_edict()
  # print(sol_Loader.total_samples)
  # for k, v in sol_Loader.ez_dataset_relations.items():
  # 	print(k)
  # for k, v in dataset.entities.items():
  # 	print(k, dataset.entities[k].vocab_size)
  # for k, v in dataset.relations.items():
  # 	print(k, dataset.relations[k].vocab_size)

  # # test run MoveLensDataLoader
  # sol_Loader=MovieLensDataLoader(batch_size=8)
  # dataset = sol_Loader.load_edict()
  # batch_sampler=sol_Loader.get_batch()
  # print(batch_sampler)
  # print(sol_Loader.total_samples)

  pass


if __name__ == '__main__':
  main()
