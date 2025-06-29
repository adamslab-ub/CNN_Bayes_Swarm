# CNN-Bayes-Swarm <br/>

In this paper, we proposed a learning based real-time down-sampling method for swarm-search mission and validated its performance on the Bayes-Swarm algorithm for several environments. We designed input, output abstractions to use a Convolutional Neural Network (CNN) to intelligently down-sample the observations.
The supplement associated with the paper can be found in the file Supplement.pdf

How to use the code:

    Training:
        The folder train_env contains the codes to train a CNN architecture for real time down-sampling. To start the training, please run the file train_CNN.py <br/>
        The implementations of the functions "Generate Matrix" and "Nearest-Points" in Algorithm 1 of our paper are in the file "environment.py". <br/>
        You can change the number of robots and customize the robot trajectories in the file 'sample_trajectories.py' <br/>

    Evaluation:
        The folder "test_env" contains the codes to evaluate the CNN down-sampler in the Bayes-Swarm algorithm in Env 2 of our paper. We have also provided the code implementations of other baseline down-samplers. 
        You can easily modify the signal source landscape in the source.py file 


Citation:

    Please cite our work if you find it useful.

Bhatt, A., Witter, J., KrishnaKumar, P., Paul, S., Chowdhury, S. "Learning-based Real-time Down-sampling for Scalable Decentralized Decision-Making in Swarm Search", Journal of Computing and Information Scince in Engineering (JCISE) (in press)