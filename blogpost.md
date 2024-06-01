## 1 Introduction

Over the past few years, deep learning research has seen significant progress in solving graph learning tasks. A crucial aspect of such problems is maintaining equivariance to transformations, such as rotations and translations, allowing for instance, to reliably predict physical properties of molecules. In this section, we will provide an overview of the predominant methods used in this domain, along with an introduction to a new method: E(n) Equivariant Simplicial Message Passing Networks (EMPSNs) [3].

Graph Neural Networks (GNNs) [13], namely their most common variant, Message Passing Neural Networks (MPNNs) [5] are instrumental for learning on graph data. Simple MPNNs however, have a number of drawbacks. Firstly, they are limited in their ability to learn higher-dimensional graph structures such as cliques (a set of points that are all connected to each other), since communication normally only happens from nodes to other nodes. Secondly, they suffer from over-smoothing; nodes of a graph iteratively update their features by aggregating the features of their neighbors, a process by which updated node features become increasingly similar. Previous works attempt to improve MPSNs’ expressivity by considering higher-dimensional simplices in the graph as learnable features [1] [8]. While these methods provide the tools for more powerful graph learning models, they do not concern themselves with equivariance.

As stated in the original EMPSN [3] paper, many real-life problems have a natural symmetry to translations, rotations, and reflections (that is, to the Euclidean group E(n)), such as object recognition or predicting molecular properties. Many approaches have been proposed to ensure E(n) equivariance: Tensor field networks [14], SE(3) Transformers [4], E(n) Equivariant Graph Neural Networks [12] among others. These works are particularly useful for working with geometric graph data, such as molecular point clouds; they use the underlying geometry of the space in which the graph is positioned to ensure E(n) equivariance. In this case, however, the lack of higher-dimensional features remains a limiting factor for the reasons stated previously. EMPSNs [3] are a novel approach to learning on geometric graphs and point clouds that is equivariant to the euclidean group E(n) (rotations, translations, and reflections). The method combines geometric and topological graph approaches to leverage both benefits. Its main contributions related to our reproduction study are the following:

1. A generalization of E(n) Equivariant Graph Neural Networks (EGNNs), which can learn
features on simplicial complexes.
2. Experiments showing that the use of higher-dimensional simplex learning improves perfor-
mance compared to EGNNs and MPSNs without requiring more parameters and proving
to be competitive with SOTA methods on the QM9 dataset [11], [10].

Additionally, their results suggest that incorporating geometric information serves as an effective
measure against over-smoothing.

In our work, we attempt to reproduce the results of the original EMPSN paper and extend the
method, rewriting parts of the author’s code to use a common suite for learning on topological
domains. The suite allows us to test how a different graph lifting procedure (an operation
that obtains higher-order simplices from graph data) compares to the one used in the original
paper.

## 2 Theoretical background

Message passing neural networks have seen an increased popularity since their introduction [5].
In this blogpost, we will elaborate on how message passing networks are adapted to work with
simplicial complexes, as proposed by [3]. We introduce the relevant definitions of message pass-
ing, simplicial complexes, equivariant message passing networks and message passing simplicial
networks.

### 2.1 Message passing
<a name="messpass"></a>
Let \( G = (V,E) \) be a graph consisting of nodes \( V \) and edges \( E \). Then let each node \( v_i \in V \) and edge \( e_{ij} \in E \) have an associated node feature \( \mathbf{f}_i \in \mathbb{R}^{c_n} \) and edge feature \( \textbf{a}_{ij} \in \mathbb{R}^{c_e} \), with dimensionality \( c_n, c_e \in \mathbb{N}_{>0} \). In message passing, nodes have hidden states (features). We update nodes' features iteratively via the following procedure:

\[
\mathbf{m}_{i j}=\phi_m\left(\mathbf{f}_i, \mathbf{f}_j, \mathbf{a}_{i j}\right)
\]

\[
\mathbf{m}_i=\underset{j \in \mathcal{N}(i)}{\operatorname{Agg}} \mathbf{m}_{i j}
\]

\[
\mathbf{f}_i^{\prime}=\phi_f\left(\mathbf{f}_i, \mathbf{m}_i\right)
\]

First, we find messages from \( v_j \) to \( v_i \) (equation \ref{compute_message}). We then aggregate messages to \( v_i \), with \( \operatorname{Agg} \) being any permutation invariant function over the neighbors (equation \ref{aggregate_messages}). Finally, we update the hidden state (features) of all nodes \( \mathbf{f}_i \) (equation \ref{update_hidden}). \( \mathcal{N}(i) \) denotes the set of neighbours of node \( v_i \), and \( \phi_m \) and \( \phi_f \) are multi-layer perceptrons. This sequence of steps is one iteration and is performed by what we call a message-passing layer.

After passing our input graph's features through several successive message-passing layers, permutation-invariant aggregation is applied to all final hidden states of the nodes in order to get a hidden state that represents the entire graph.

### 2.2 Simplicial complexes

A simplex in Geometry is the extension of the concept of triangles to multiple dimensions. An
n-simplex consists of n+1 fully connected points, i.e. a 0-simplex is a point, a 1-simplex is a line,
a 2-simplex is a triangle, a 3-simplex is a tetrahedron, a 4-simplex is a 5-cell, etc. In an article
entitled Architectures of Topological Deep Learning: A Survey of Message-Passing Topological
In Neural Networks [9], simplices are mainly referred to as cells of rank r (r-cells for short),
where for example, a 1-simplex is a 1-cell. To assign features to higher-dimensional simplices
in graphs, we turn to the definition of a generalized notion of graphs called abstract simplicial
complexes.