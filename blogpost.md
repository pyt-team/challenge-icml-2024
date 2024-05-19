<!-----

You have some errors, warnings, or alerts. If you are using reckless mode, turn it off to see inline alerts.
* ERRORs: 0
* WARNINGs: 1
* ALERTS: 1

Conversion time: 0.688 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β36
* Sun May 19 2024 13:50:15 GMT-0700 (PDT)
* Source doc: blogpost.md

WARNING:
You have some equations: look for ">>>>>  gd2md-html alert:  equation..." in output.

----->


<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 0; WARNINGs: 1; ALERTS: 1.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p><a href="#gdcalert1">alert1</a>

<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>




* _Introduction: An analysis of the paper and its key components. Think about it as a nicely formatted review as you would see on OpenReview.net. It should contain one paragraph of related work as well._
    * combination of geometric information with topological information in message passing framework
    * Very expensive to train (and inference)
    * Vietoris rips lift has some locality assumptions that are ungrounded
    * related work (EGNN and other message passing papers)
* Background:
    * simplicial complexes
    * message passing
    * message passing in simplicial complexes (?)
* _Exposition of its weaknesses/strengths/potential which triggered your group to come up with a response._
    * Strengths:
        * Combination of equivariant geometric information with topological information is interesting in graph neural network is interesting
        * performance on QM9 shows potential of the proposed methodology
    * weaknesses:
        * no message passing between 2nd order simplexes. 
        * readability of provided code is subpar and difficult to comprehend
        * results are difficult to reproduce and bugs are present within the code as provided by the author.
* _Describe your novel contribution._
    * Refactoring of the codebase to use TopoX as much as possible. TopoX is the salient topological deep learning library for Python. 
* _Results of your work (link that part with the code in the jupyter notebook)_
* _Conclude_
* _Close the notebook with a description of each student's contribution._

Introduction

<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>



The original paper proposes a novel approach to learning on geometric graphs and point clouds that is equivariant to the euclidean group E(n) (rotations, translations, and reflections); the method is called Equivariant Message Passing Simplicial Networks (EMPSNs). This work uses the most common type of Graph Neural Networks (GNNs)[^1]: Message Passing Neural Networks (MPNNs) for graph learning tasks where equivariance is important (e.g. molecular property prediction). Message passing neural networks (MPNNs) have a number of drawbacks: Firstly the MPNNs are limited in learning higher dimensional graph structures such as cliques[^2]. Secondly, it is well known deep message passing neural networks can cause over-smoothing, after many local neighborhood aggregations,  features of all nodes converge to the same value leading to indistinguishable node representations. EMPSN is designed to combat both of these well-known drawbacks of message passing neural networks. The idea of EMPSNs is therefore to consider a more elaborate topological space by incorporating a graph’s higher dimensional simplexes as learnable features.

Imagine a molecular graph, the molecule as a whole has a number of quantifiable properties and each atom (node) has geometric information. In EMPSN not only do the nodes pass messages to one another but simplexes created from the node in the graph also pass messages to one another. These simplexes have E(n) equivariant geometric features information derived from the geometric information of the nodes contained within this simplex. In this way, higher dimensional graph structures are explicitly represented in the message passing framework. The results of the paper suggest that this method also combats over-smoothing (explain further).

EMPSNs are applicable to geometric graph data and can be used to predict graph level features. The methodology starts by performing a graph lift, To limit the amount of simplexes constructed a Vietoris-Rips lift is applied. The simplexes this yields are added as nodes to the graph, at this point the simplexes are purely topological objects, a geometric realization is achieved by calculating a number of E(n) equivariant features based on the geometric information of the nodes contained in the simplex. Examples are, relative distance of nodes, angles between nodes and volume of the simplex. The higher order the simplex the more features are able to be calculated. 

After this step the molecule is enriched with geometrically embedded higher order structures in the form of line segments and triangles. In the training of the network, messages are passed between nodes and between simplexes according to a hardcoded schema. 0 dimensional simplexes communicate with 0 dimensional and 1 dimensional simplexes. 1 dimensional simplexes communicate with 1 dimensional simplexes and 2 dimensional simplexes. 2 dimensional simplexes only communicate with 1 dimensional simplexes.

Related work

This methodology is novel since it is the only work that combines simplicial complexes with E(n) equivariant message passing in GNNs. There two branches of Graph neural network research that is closely related to the paper in question. Firstly, Message passing simplical networks (MPSNs). This methodology also enriches the topology of the input graphs with simplexes the difference with this work is that the feature information of the added simplexes is not necessarily equivariant under any transformation group.[^3] Another related branch of work are Equivariant Graph neural networks, one paper in particular is mentioned in the EMPSN paper. In this paper, the weighted sum of all of relative differences is used in multiple steps of the message passing to make the node updates E(n) equivariant.

EMPSN essentially borrowed ideas from both branches to construct a new methodology.

We acknowledge that the choice for topological complex is an important one since it is an assumption of shape of the higher order structure in the data. 

 \



## Strengths and weaknesses

An immediate strength of the proposed graph neural network is the utilization of the invariant geometric framework, combined with the topological information that is generated through the lifting operation, leads to a very data efficient learning environment with comparable results to SotA methods. This strength can immediately be seen in the method’s performance on the QM9 dataset. Even though the current method is general and does not utilize a large network. it is comparable to, or even outperforms more specific methods (that utilize domain-specific strategies) on certain features. 

Another strength is the general applicability of this method. The proposed equivariant simplicial message passing network is in theory applicable to any dataset, since we do the lifting operation, although likely is that data that has a certain structure will utilize its effects better. Added to this is that it scales more efficiently to data (it scales based on the simplicial structures and their depth).

Finally, as mentioned before, it appears empirically that this method performs decent, even when the higher order simplicial structures aren’t being utilized much, and it also is good at preventing over-smoothing, a common problem in GNN’s.


# The current implementation also has a couple practical and theoretical weaknesses. 

practically, the readability of the code was very hard, without explanation and some random hardcoded parts that should not be, notwithstanding the bugs that were found. Weird design choices using strings that seem illogical are used. There is a lot of experimentation going on with 

The original paper does two experiments, one using an Invariant MPSN and one that uses the full Equivariant MPSN. The code for the latter unfortunately isn’t included in the paper so there is no way of reproducing. The former hardcoded the potential invariances (e.g. from rank 0->1, there’s 3), but there is no explanation what this number is based on, neither is it clear how this number increases into higher level simplicial structures, i.e. what would be the rank 2->3 invariances? The paper lacks justification for many design choices in this regard.

Another justification that isn’t clear or explained, is what the ‘search range’ (Delta) is that the Vietoris-Rips algorithm uses. 

Besides, the original papers results seem to be too good, the reproduction attempts with the original code do show very similar trends, but they don’t seem to score as high as accurately as the original paper seems to imply it should. This could be a coincidence as it hasn’t been tested over several seeds due to computation cost.

A design choice that isn’t clear is how ‘high’ one should go with the simplicial structures and their communication. For the current paper it was chosen that the communication between simplexes only was over the same rank ones and rank +1. As an exception however, 2nd order simplexes are not allowed to communicate amongst each other. It is unclear if this choice is grounded in empirical results, nor is it clear what the ‘ideal’ height should be of the lifting and corresponding message passing. One could for example not only allow communication upward, but also downward, or go to the 4 dimensional level and see if that garners better results. All of these are likely also dataset dependent.




## Describe your novel contribution.

For our contribution, we decided to attempt to reformat the code to adhere to the new TopoX framework[^4]. TopoX is a topological deep learning suite built on top of pytorch and it is meant to create a general framework and provide building blocks for geometric and topological deep learning applications. The current paper is a prime candidate to utilize this framework. Besides, TopoX creators have placed a request for people to translate lifting operations into their framework. Both these additions are qualitatively useful additions to not only the current paper, streamlining and clarifying the procedures, but also the TopoX suite, by increasing code written by their standards.


## Results of your work (link that part with the code in the jupyter notebook)

our code is located in this forked repository: [https://github.com/martin-carrasco/challenge-icml-2024.git](https://github.com/martin-carrasco/challenge-icml-2024.git) 

As a summary:



    * We have reformatted the code for the Vietoris-Rips lift.
    * We have attempted to restructure the codebase as much as possible, using the building blocks of TopoX, specifically the ones for simplicial layers and the message passing framework. By doing so we have changed the input required for the layers from the indexing-based structure that the original authors used, into a adjacency/incidence matrix format, which is a more generalizable structure. 
    * The main changes are in the lifting subfolder and in the models subfolder.

## 
        Conclusion


In this post we have investigated a novel ‘proof of concept’ study that investigates the use of simplicial structures combined with geometric properties as a useful avenue to study. the paper designs an equivariant or invariant message passing simplicial network. It seems that the author is fair in his assessment of the utility. It is a compact network and the usage of invariance combined with simplices seems prudent, although the accuracy of the predictions seems to not reach the same level as the original paper. We name a couple of found strengths and weaknesses, focussing on the efficiency and generalisability. While also remarking the difficulty of the code and its implementation. By refitting the code into a promising new suite for topological deep learning, named TopoX, we hope to make the code more accessible and to contribute to the growing amount of papers using TopoX. \
 \



## Contributions

Jesse Wonnink

Martin Carrasco

Nordin Belkacemi

Andreas Berentzen

Valeria Sepicacchi


<!-- Footnotes themselves at the bottom. -->
## Notes

[^1]:
     F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner and G. Monfardini, "The Graph Neural Network Model," in IEEE Transactions on Neural Networks, vol. 20, no. 1, pp. 61-80, Jan. 2009, doi: 10.1109/TNN.2008.2005605.
    keywords: {Neural networks;Biological system modeling;Data engineering;Computer vision;Chemistry;Biology;Pattern recognition;Data mining;Supervised learning;Parameter estimation;Graphical domains;graph neural networks (GNNs);graph processing;recursive neural networks},

[^2]:
     
    Garg, V., Jegelka, S. &amp; Jaakkola, T.. (2020). Generalization and Representational Limits of Graph Neural Networks. <i>Proceedings of the 37th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 119:3419-3430 Available from https://proceedings.mlr.press/v119/garg20c.html.

[^3]:
    Bodnar, C., Frasca, F., Wang, Y., Otter, N., Montufar, G.F., Lió, P. &amp; Bronstein, M.. (2021). Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks. <i>Proceedings of the 38th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 139:1026-1037 Available from https://proceedings.mlr.press/v139/bodnar21a.html.

[^4]:
     [[2402.02441] TopoX: A Suite of Python Packages for Machine Learning on Topological Domains](https://arxiv.org/abs/2402.02441) 

