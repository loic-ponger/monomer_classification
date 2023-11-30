#! /usr/bin/python3.9

import argparse
import tempfile 
import os
import csv
import timeit
import collections
from random import sample 
import re
import io

import pandas as pd
import numpy as np
import random as rd
from pysam import FastaFile
import pyhmmer
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 

import scipy.cluster.hierarchy as shc
import seaborn as sns
import matplotlib.pyplot as plt

from Bio.Align.Applications import MuscleCommandline
from Bio import SeqIO
from Bio import Phylo, AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from ete3 import Tree
import dendropy


###################################
def read_kmers(infile, verbose=0):
	if verbose > 5:
		print("  reading kmer file %s..." % (infile))
	kmers=pd.read_csv(infile, delimiter=" ")
	if verbose > 5:
		print("    -> %d sequences" % (kmers.shape[0]))
		print("    -> %d kmers" % (kmers.shape[1]-1))
	return(kmers)
###################################
def read_fasta(infile, verbose=0):
	if verbose > 5:
		print("  reading fasta file %s ..." % (infile))
	sequences=FastaFile(infile)
	if verbose > 5:
		print("    -> %d sequences" % (len(sequences)))
	return(sequences)
###################################
def write_fasta(sequences, outfile=None, verbose=0):
	if outfile == None:
		outfile = tempfile.NamedTemporaryFile().name	
	with open(outfile, 'w') as f:
		nb = 0
		for seq in sequences:
			f.write(">seq"+str(nb)+"\n")
			f.write(seq+"\n")
			nb = nb+1
	f.close()
	if verbose > 5:
		print("Sequences written in %s (%d seq)" % (outfile, len(sequences)))
	return(outfile)
###################################
def write_fasta_for_all_consensus(all_consensus, outfile=None, verbose=0):
	if outfile == None:
		outfile = tempfile.NamedTemporaryFile().name	
	with open(outfile, 'w') as f:
		nb = 0
		for clust in all_consensus:
			for seq in all_consensus[clust]:
				if verbose > 6:
					print("clust cons: %d -> %s" % (clust, seq))
				f.write(">consensus_"+str(clust)+"_"+str(nb)+"\n")
				f.write(str(seq)+"\n")
				nb = nb+1
	f.close()
	if verbose > 5:
		print("         -> %d sequences written in %s" % (nb, outfile))
	return(outfile)
###################################
def scale_kmers(kmers, verbose=0):
	if verbose > 5:
		print("  scaling kmers ...")
	kmers_scaled = StandardScaler().fit_transform(kmers.iloc[:,1:])
	if verbose > 5:
		print("    -> %d sequences" % (kmers_scaled.shape[0]))
		print("    -> %d kmers" % (kmers_scaled.shape[1]))
	return(kmers_scaled)
	
###################################
def my_PCA(scaled_kmers, nb_PC=10, verbose=0):
	if verbose > 5:
		print("  computing principal components (nb=%d)..." % (nb_PC))
	pca = PCA(n_components=nb_PC)
	principalComponents = pca.fit_transform(kmers_scaled)
	if verbose > 5:
		print("   -> explained variance:")
		print(pca.explained_variance_ratio_)
	return(principalComponents)
	
###################################
def my_scaled_PCA(all_kmers, nb_seq=None, nb_PC=10, verbose=0):
	if verbose > 5:
		print("  computing scaled principal components (nb=%d)..." % (nb_PC))
		
	all_kmers.reset_index(drop=True, inplace=True)
	kmers_scaled = scale_kmers(all_kmers, verbose=verbose)
	
	if nb_seq == None or kmers_scaled.shape[0] <= nb_seq:
		sample_kmers_scaled = kmers_scaled
	else:
		sample_kmers_scaled = kmers_scaled.sample(n=nb_seq)

	pca = PCA(n_components=nb_PC)
	pca.fit_transform(sample_kmers_scaled)
	
	print(sample_kmers_scaled.shape)
	print(kmers_scaled.shape)
	principalComponents = pca.transform(kmers_scaled)
	print(principalComponents.shape)
	principalComponents = pd.DataFrame(data = principalComponents)
	principalComponents.reset_index(drop=True, inplace=True)

	if verbose > 5:
		print("   -> explained variance:")
		print(pca.explained_variance_ratio_)
		
	finalDf = pd.concat([all_kmers.iloc[:,0], principalComponents], axis = 1)
			
	return(finalDf)

###################################
## Not used anymore
def reformat_PCA_output(principalComponents, seq_names, verbose=0):
	if verbose > 5:
		print("  reformating component table...")
	finalDf = pd.concat([seq_names, pd.DataFrame(data = principalComponents)], axis = 1)
	if verbose > 5:
		print("   -> %d sequences" % (finalDf.shape[0]))
		print("   -> %d components"     % (finalDf.shape[1]-1))
	return(finalDf)
	
###################################
def plot_PCA_plan(finalDf, ax1=1, ax2=2, verbose=0):
	sns.scatterplot(x=finalDf.iloc[:,ax1], y=finalDf.iloc[:,ax2])
	plt.show()
	
###################################
def plot_PCA_plan_with_clustering(finalDf, clustering, ax1=1, ax2=2, verbose=0):
	sns.scatterplot(x=finalDf.iloc[:,ax1], y=finalDf.iloc[:,ax2], hue=clustering, palette="rainbow")
	plt.show()

###################################
def plot_dendrogram(dendro, verbose=0):
	shc.dendrogram(Z=dendro)
	plt.show()

###################################
def my_LDA_training(x, y, verbose=0):
	if verbose > 5 :
		print("  training the lda ...")
	model = LinearDiscriminantAnalysis()
	model.fit(x, y)
	return(model)

###################################
def my_LDA_prediction(model, new_x, verbose=0):
	if verbose > 5 :
		print("  predicting with the lda ...")
		print(new_x.shape)
	pred = model.predict(new_x)
	if verbose > 5 :
		unq,counts = np.unique(pred,return_counts=True)
		print ("   -> cluster names:")
		print(unq)
		print ("   -> cluster counts:")
		print(counts)
	return(pred)
	
###################################
def cluster(finalDf, nb_clust, verbose=0):
	if verbose > 5:
		print("  clustering ...")
	clustering_model = AgglomerativeClustering(n_clusters=nb_clust, affinity='euclidean', linkage='ward')
	clustering_model.fit(finalDf)
	if verbose > 5 :
		unq,counts = np.unique(clustering_model.labels_,return_counts=True)
		print (unq)
		print(counts)
	return(clustering_model)
	
###################################
def	find_new_clusters(previous_clusters, next_clusters, verbose=0):
	cluster_matchs = [ii[0] for ii in list(set(zip(previous_clusters,next_clusters)))]
	counts = collections.Counter(cluster_matchs)
	cluster = sorted(cluster_matchs, key=counts.get, reverse=True)[0]
	new_clusters = [ii[1] for ii in list(set(zip(previous_clusters,next_clusters))) if ii[0]==cluster ]
	return(new_clusters)
	
###################################
def	cluster_validation_by_phylogeny(sequences, sequence_names, cluster_names, selected_clusters, nb_seq_for_consensus=100, nb_of_consensus=10,  threshold_for_monophyly=0.9,  verbose=0):
	all_consensus = {}
	for clust in selected_clusters:
		selected_sequence_names = [na for na,cl in zip(sequence_names,cluster_names) if cl==clust]
		if verbose > 7:
			print(" computing consensus for cluster %d (%d seq)" % (clust,len(selected_sequence_names)) )
	
		splited_sequences_names = split_sequence_names(selected_sequence_names, nb_seq_for_consensus, nb_of_consensus, verbose=verbose)
		
		
		
		
		#consensus_from_sequence_names(splited_sequences_names[0], sequences,  verbose=verbose)
		all_consensus[clust] = list(map(lambda ssn: consensus_from_sequence_names(ssn, sequences,  verbose=verbose), splited_sequences_names))
		if verbose > 7:
			print("     -> %d consensus" % (len(all_consensus[clust])))
			print("     -> %d total consensus" % (len(all_consensus)))
			print("######")
	all_consensus_file = write_fasta_for_all_consensus(all_consensus, verbose=verbose)
	all_consensus_alignment_file = muscle_alignment(all_consensus_file, outfile=None, binary="muscle", verbose=verbose)
	align = AlignIO.read(all_consensus_alignment_file,'fasta')
	calculator = DistanceCalculator('identity')
	distMatrix = calculator.get_distance(align)
	constructor = DistanceTreeConstructor()
	NJTree = constructor.nj(distMatrix)
	if verbose > 1:
		Phylo.draw_ascii(NJTree)
	is_valid, props = check_monophilies_for_two_groups(NJTree, threshold_for_monophyly, verbose)
	
	return(is_valid, props)
###################################
def get_all_names_clusters(tree, verbose=0):
	tip_names=[]
	cluster_names=[]
	for leaf in tree.get_terminals(): 
		tip_names.append(leaf.name)
		cluster_names.append(re.sub("_.*", "", re.sub("^[^_]+_", "", leaf.name)))
	return(tip_names, cluster_names)
	
###################################
def convert_PhyloTree2dendroTree(phylo_tree, verbose=9):
	string_tree = io.StringIO()
	Phylo.write(phylo_tree, string_tree, "newick")
	string_tree.seek(0)
	dendro_tree = dendropy.Tree.get(file=string_tree, schema='newick', preserve_underscores=True)	
	return(dendro_tree)
###################################
def convert_PhyloTree2ete3Tree(phylo_tree, verbose=9):
	string_tree = io.StringIO()
	Phylo.write(phylo_tree, string_tree, "newick")
	string_tree.seek(0)
	ete3_tree = Tree(string_tree.read(), format=1)	
	ete3_tree.unroot()
	return(ete3_tree)
###################################
def check_monophilies_for_two_groups(dendro_tree, threshold=0.9, verbose=0):
	max_prop0=0
	min_prop0=100
	max_prop1=0
	min_prop1=100
	
	ete3tree = convert_PhyloTree2ete3Tree(dendro_tree, verbose)
	all_clusters_names=[  get_cluster_name(item) for item in ete3tree.get_leaf_names()  ]
	all_clusters_counts = dict(Counter(all_clusters_names).items())
	for node in ete3tree.iter_descendants("postorder"):
		node_clusters_names=[  get_cluster_name(item) for item in node.get_leaf_names()  ]
		node_clusters_counts = dict(Counter(node_clusters_names).items())

		cluster_names = list(all_clusters_counts.keys())
		prop0 = node_clusters_counts.get(cluster_names[0], 0) / all_clusters_counts[cluster_names[0]]
		prop1 = node_clusters_counts.get(cluster_names[1], 0) / all_clusters_counts[cluster_names[1]]
		if prop0 > max_prop0:
			max_prop0=prop0
		if prop0 < 	min_prop0:
			min_prop0=prop0
		if prop1 > max_prop1:
			max_prop1=prop1
		if prop1 < 	min_prop1:
			min_prop1=prop1
		
		#print([prop0, prop1])
		if (prop0 >= threshold and prop1 <= (1-threshold)) or (prop1 >= threshold and prop0 <= (1-threshold)):
			if verbose > 6:
				print("cluster %s: %f (threshold %f)" % ( cluster_names[0],prop0,  threshold))
				print("cluster %s: %f (threshold %f)" % ( cluster_names[1],prop1,  threshold))
			return(True, [max_prop0, min_prop0, max_prop1, min_prop1])	
		
	return(False, [max_prop0, min_prop0, max_prop1, min_prop1])
		
###################################
def get_cluster_name(my_name, verbose=0):
	return( re.sub("_.*", "", re.sub("^[^_]+_", "", my_name)))
		
###################################
def	best_HCA_clustering(pc_table, sequences, nb_seq=None, nb_cluster_min=2, nb_cluster_max=None, min_cluster_size=1, 
            nb_seq_for_consensus=100, nb_of_consensus=10, threshold_for_monophyly=0.9, verbose=0):
	sequence_names=pc_table.iloc[:,0]
	best_clusters=[0] * len(sequence_names)
	
	if nb_seq == None or pc_table.shape[0] <= nb_seq:
		sample_pc_table = pc_table
	else:
		sample_pc_table = pc_table.sample(n=nb_seq)
			
	if nb_cluster_max == None or nb_cluster_max < nb_cluster_min:
		nb_cluster_max =  sample_pc_table.shape[0]
	ward_clustering = shc.linkage(sample_pc_table.iloc[:,1:], method="ward", metric="euclidean")
	previous_clusters = best_clusters
	ncl=nb_cluster_min
	continue_spliting=True
	while continue_spliting == True:
			print ("###############")
			print(ncl)
			cluster_labels = shc.cut_tree(ward_clustering, n_clusters=ncl).reshape(-1, )
			unq,counts = np.unique(cluster_labels,return_counts=True)
			print (unq)
			print(counts)
			
			new_clusters = find_new_clusters(previous_clusters, cluster_labels, verbose=args.verbose)
			previous_clusters = cluster_labels
			print("---------")
			print(new_clusters)
		
			my_LDA = my_LDA_training(sample_pc_table.iloc[:,1:], cluster_labels, verbose=args.verbose)
			pc_table_prediction = my_LDA_prediction(my_LDA, pc_table.iloc[:,1:], verbose=args.verbose)
			unq,counts = np.unique(pc_table_prediction,return_counts=True)
			print (unq)
			print(counts)
		
			is_valid, props = cluster_validation_by_phylogeny(sequences, sequence_names=sequence_names, cluster_names=pc_table_prediction, selected_clusters=new_clusters, nb_seq_for_consensus=nb_seq_for_consensus, nb_of_consensus=nb_of_consensus, threshold_for_monophyly=threshold_for_monophyly, verbose=verbose )

			ncl = ncl + 1
			if ncl <= nb_cluster_max and min(counts) >= min_cluster_size and is_valid == True:
				print("continuing !")
				print("%d <= %d ?" % (ncl,nb_cluster_max))
				print("%d >= %d ?" % (min(counts),min_cluster_size))
				print("%s is valid ?" % (str(is_valid)))
				best_clusters = pc_table_prediction
			else:
				print("stooooooooooooping !")
				print("%d <= %d ?" % (ncl,nb_cluster_max))
				print("%d >= %d ?" % (min(counts),min_cluster_size))
				print("%s is valid ?" % (str(is_valid)))
				continue_spliting = False
	
	unq,counts = np.unique(best_clusters,return_counts=True)
	#print (unq)
	#print(counts)
	return([sequence_names, best_clusters])
	
	
###################################
def my_HCA(data, verbose=0):
	if verbose > 5:
		print("  building dendrogram...")
	dendro = shc.linkage(data, 
            method='ward', 
            metric="euclidean")	
	return(dendro)

###################################
def muscle_alignment(infile, outfile=None, binary="muscle", verbose=0):
	if outfile == None:
		outfile = tempfile.NamedTemporaryFile().name
	muscle_cline = MuscleCommandline(binary, input=infile, out=outfile)
	stdout, stderr = muscle_cline()
	if verbose > 7:
		print("   alignment ...")
		print("       ->   input file: %s" % (infile))
		print("       ->   output file: %s" % (outfile))
	return(outfile)

###################################
def consensus_from_alignment(alignment_file, consensus_file=None, model_name=None, binary_hmmbuild="hmmbuild", binary_hmmemit="hmmemit", verbose=0):
	if verbose > 7:
		print("  building consensus...")
	if model_name == None:
			model_name = "tmp"
	if consensus_file == None:
		consensus_file = tempfile.NamedTemporaryFile().name
	hmm_file = tempfile.NamedTemporaryFile().name		

	build_command = binary_hmmbuild + " -n " + model_name + " " + hmm_file + " " + alignment_file + " 1> /dev/null 2> /dev/null"
	emit_command = binary_hmmemit + " -c " +  hmm_file + " 1>" + consensus_file + " 2> /dev/null"
	if (verbose > 7):
		print("       -> building command: %s" % (build_command))
	os.system(build_command )
	if (verbose > 7):
		print("       -> emiting command: %s" % (emit_command))
	os.system(emit_command )
	return(consensus_file)

###################################
def consensus_from_sequences(sequences, binary_muscle="muscle", binary_hmmbuild="hmmbuild", binary_hmmemit="hmmemit", verbose=0):
	sequence_file = write_fasta(sequences, verbose=verbose)
	alignment_file = muscle_alignment(sequence_file, verbose=verbose)
	consensus_file = consensus_from_alignment(alignment_file, verbose=verbose)
	return(consensus_file)
	
	
###################################
def split_sequence_names(list_of_names,  nb_seq=100, nb_of_splits=None, random=True, remove_incomplete_subsets=True, verbose=0):
		if verbose > 7:
			print("   spliting sequence list into subsets (%d seq, % subset length)" % (len(list_of_names), nb_seq))
		if random == True:
			list_of_names = sample(list_of_names, len(list_of_names))
		subsets = [list_of_names[i:i+nb_seq] for i in range(0, len(list_of_names), nb_seq)]
		if verbose > 7:
			print("     -> %d subsets" % (len(subsets)))

		if remove_incomplete_subsets == True:
			if len(subsets[-1]) < nb_seq:
				del subsets[-1]
			if verbose > 7:
				print("     -> %d complete subsets" % (len(subsets)))
		if nb_of_splits != None:
			return(subsets[0:nb_of_splits])
		return(subsets)
		
###################################		
def consensus_from_sequence_names(sequence_names, pysam_iterator, verbose=0):
	selected_sequences = [ pysam_iterator.fetch(seq) for seq in sequence_names ]
	consensus_file = consensus_from_sequences(selected_sequences, verbose=verbose)
	consensus_record = list(SeqIO.parse(open(consensus_file, mode='r'), 'fasta'))[0]
	return(consensus_record.seq)
	
###################################
def check_arguments(args, verbose=0):
	if args.nb_clust_min < 2:
		print("Warning: minimal number of clusters should be an interger greater than 1")
		args.nb_clust_min = 2
	
	if args.nb_clust_max != None and args.nb_clust_max < 2:
		print("Warning: maximal number of clusters should be an interger greater than 1 or None")
		args.nb_clust_max = None
		
	if args.output == None:
		args.output = args.kmer_file + "_prediction.dat"
		
	args.min_cluster_size = max(args.min_nb_of_consensus * args.nb_seq_for_consensus, args.min_cluster_size)
	
	return(args)
###################################
def write_clustering(sequence_names, cluster_names, outfile, verbose=0):
	with open(outfile, 'w') as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerows(zip(['sequence'], ['cluster']))
		writer.writerows(zip(sequence_names,cluster_names))	
	return(0)
###################################
def my_HCA_binary(pc_table, nb_seq=None, verbose=0):
	sequence_names=pc_table.iloc[:,0]
	best_clusters=[0] * len(sequence_names)
	#sampling of the table if needed
	if nb_seq == None or pc_table.shape[0] <= nb_seq:
		sample_pc_table = pc_table
	else:
		sample_pc_table = pc_table.sample(n=nb_seq)		
	# binary HCA for the sampled sequences		
	#print(	sample_pc_table.head)
	
	#print(	sample_pc_table.shape)
	ward_clustering = shc.linkage(sample_pc_table.iloc[:,1:], method="ward", metric="euclidean")
	cluster_labels = shc.cut_tree(ward_clustering, n_clusters=2).reshape(-1, )	
	# prediction for all the sequences
	my_LDA = my_LDA_training(sample_pc_table.iloc[:,1:], cluster_labels, verbose=args.verbose)
	pc_table_prediction = my_LDA_prediction(my_LDA, pc_table.iloc[:,1:], verbose=args.verbose)
	return(sequence_names, pc_table_prediction)
###################################	
def	convert_dict2lists(clust_dict, verbose=0):
	seq_names = []
	clust_names = []
	for kk in clust_dict:
		#print("key: %d" % (kk))
		clust_names.extend([kk] * len(clust_dict[kk]))
		seq_names.extend(clust_dict[kk])
	#print(clust_names)
	#print(len(clust_names))
	#print(len(seq_names))
	#print(len(clust_dict))
	return(seq_names, clust_names)
	
	
###################################
def best_HCAbinary_clustering(kmers, 
						sequences, 
						nb_seq_PCA=None, 
						nb_PC=None, 
						nb_seq_HCA=None, 
						nb_cluster_min=2, 
						nb_cluster_max=None, 
						min_cluster_size=1, 
						nb_seq_for_consensus=100, 
						nb_of_consensus=10, 
						threshold_for_monophyly=0.9, 
						verbose=0):
		seq_names=kmers.iloc[:,0]
		clusters=[0] * len(seq_names)
		list_of_clusters=[0]
		nb_clust_max=0
		clust_dict = {}
		clust_dict[0] = kmers.iloc[:,0]
		
		for clust in list_of_clusters:
			tmp_kmers=kmers.loc[kmers.iloc[:,0].isin(clust_dict[clust])]
			tmp_PC = my_scaled_PCA(tmp_kmers, nb_seq=nb_seq_PCA, nb_PC=nb_PC, verbose=verbose)
			#tmp_PC.dropna(inplace=True)
			tmp_HCA_seq_names,tmp_HCA_cluster_names = my_HCA_binary(tmp_PC, nb_seq=nb_seq_HCA, verbose=verbose)
			a,x = np.unique(tmp_HCA_cluster_names,return_counts=True)
			if verbose > 0:
				print("Cluster %d -> kmers: %d-%d pc: %d-%d" % (clust, 
									tmp_kmers.shape[0], 
									tmp_kmers.shape[1], 
									tmp_PC.shape[0], 
									tmp_PC.shape[1]))
			#print(min(x))
			#print(min_cluster_size)
			#print(x)
			if 	min_cluster_size == None or min(x) >= min_cluster_size:			
				tmp_is_valid, props = cluster_validation_by_phylogeny(sequences, 
						sequence_names=tmp_HCA_seq_names, 
						cluster_names=tmp_HCA_cluster_names, 
						selected_clusters=[0, 1], 
						nb_seq_for_consensus=nb_seq_for_consensus, 
						nb_of_consensus=nb_of_consensus, 
						threshold_for_monophyly=threshold_for_monophyly, 
						verbose=verbose )
				#print(props)
				if tmp_is_valid== True:
					if verbose > 0:
						for nm,cl in zip(a,x):
							print("  valid clust %d (%d) -> %d seqs" % (nm,sum(x),cl))
	

					nb_clust_max = nb_clust_max + 1
					clust_dict[clust] = tmp_HCA_seq_names[tmp_HCA_cluster_names == 0]
					clust_dict[nb_clust_max] = tmp_HCA_seq_names[tmp_HCA_cluster_names == 1]
					#print(len(clust_dict[clust]))
					#print(len(clust_dict[nb_clust_max]))
					list_of_clusters.append(nb_clust_max)
					list_of_clusters.append(clust)
				else:
					if verbose > 0:
						for nm,cl in zip(a,x):
							print(" INvalid phylo   clust %d -> %d seqs" % (nm,cl))

			else:
				if verbose > 0:
					print(" INvalid clust size  clust %d -> %d seqs (min size: %d)" % (nm,cl, min(x)))


	
		#print(len(clust_dict))
		#print(clust_dict)
		seq_names, clusters = convert_dict2lists(clust_dict, verbose)
		return(seq_names, clusters)





















###################################

parser = argparse.ArgumentParser(description='')
parser.add_argument('-k', dest="kmer_file",  required=False, default="xxx.kmers5_30000_10_25_prefix_prediction.dat_fam1.fst.kmers5_30000_10_25_prefix_prediction.dat_fam10.fst.kmers5")
parser.add_argument('-s', dest="fasta_file", required=False, default="xxx")
parser.add_argument('-c', dest="nb_seq_for_consensus", type=int, default=100)
parser.add_argument('-C', dest="nb_of_consensus", type=int, default=10)
parser.add_argument('-l', dest="min_nb_of_consensus", type=int, default=2)
parser.add_argument('-T', dest="threshold_for_monophyly", type=float, default=0.9)
parser.add_argument('-P', dest="nb_seq_for_PCA", type=int, default=None)
parser.add_argument('-H', dest="nb_seq_for_HCA", type=int, default=1000)
parser.add_argument('-p', dest="nb_PC", type=int, default=10)
parser.add_argument('-m', dest="nb_clust_min", type=int, default=2)
parser.add_argument('-M', dest="nb_clust_max", type=int, default=None)
parser.add_argument('-j', dest="min_cluster_size", type=int, default=1)
parser.add_argument('-o', dest="output", type=str, default=None)
parser.add_argument('-x', dest="method", type=str, default="one_shoot")

parser.add_argument('-v', dest="verbose",   type=int, default=0)

args = parser.parse_args()
args=check_arguments(args)




kmers=read_kmers(args.kmer_file, verbose=args.verbose)
sequences=read_fasta(args.fasta_file, verbose=args.verbose)


if args.method == "one_shoot":
	###################################
	## PCA
	## scaled PCA and reformating
	myPC = my_scaled_PCA(kmers, nb_seq=args.nb_seq_for_PCA, nb_PC=args.nb_PC, verbose=args.verbose)

	##################################
	## HCA -custering
	sequence_names, best_clustering = best_HCA_clustering(myPC,
					sequences,
					nb_seq=args.nb_seq_for_HCA,
                    nb_cluster_min=args.nb_clust_min, 
                    nb_cluster_max=args.nb_clust_max, 
                    min_cluster_size=args.min_cluster_size,
                    nb_seq_for_consensus=args.nb_seq_for_consensus,
                    nb_of_consensus=args.nb_of_consensus,
                    threshold_for_monophyly=args.threshold_for_monophyly,
                    verbose=args.verbose)

	write_clustering(sequence_names, best_clustering, outfile=args.output, verbose=args.verbose)

if args.method == "shoot_again":
	sequence_names, best_clustering = best_HCAbinary_clustering(
					kmers, 
					sequences, 
					nb_seq_PCA=args.nb_seq_for_PCA,
					nb_seq_HCA=args.nb_seq_for_HCA,
					nb_cluster_min=args.nb_clust_min, 
                    nb_cluster_max=args.nb_clust_max, 
                    min_cluster_size=args.min_cluster_size,
                    nb_seq_for_consensus=args.nb_seq_for_consensus,
                    nb_of_consensus=args.nb_of_consensus,
                    threshold_for_monophyly=args.threshold_for_monophyly,
					verbose=args.verbose)
	a,x = np.unique(best_clustering,return_counts=True)
	print(a)
	print(x)
	write_clustering(sequence_names, best_clustering, outfile=args.output, verbose=args.verbose)


