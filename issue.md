avivko:
I benchmarked the FastPLM implementations of ESM2 (650M), ESMC (600M), and E1 (600M) vs the native implementation of ESMC 600M from EvolutionaryScale on an internal dataset of mine using a simple linear SVM to perform a classification task. Just to give a bit of information: the task has two different labeling granularities: one is a 4-class, and the other is a 10-class classification. I have split the data into 10 cross-validation splits for each homology-based threshold and trained a model for each of these (meaning these results are statistically less sensitive to exact splits or classification model parameters).

It seems like the FastPLMs ESMC implementation performs worse than the native one on average (each dot is the average of 10 CV splits). I unfortunately can't provide you with the data, but based on this, I would suggest you run your own benchmark to ensure parity with the base models.

<img width="447" height="706" alt="Image" src="https://github.com/user-attachments/assets/a7a8f9c1-84be-4b86-9829-dfad9a9768d0" />

lhallee:
Hey @avivko ,

Thanks for opening an issue. Could you confirm which hardware and python version / packages this was produced on?

The newest versions of Transformers (v5.0+) randomly initialize the weights of FastPLM models upon loading for some reason. This would explain the performance issue. We recommend transformers==4.57.6 as shown in our requirements.txt. Once Transformers v5 is more stable we will migrate.  

That being said, if you have the correct version of transformers, I'm not sure what the issue is. We'll have to see if the tests pass on your machine.
Best,
Logan

avivko:
Thanks for the quick response @lhallee !

I used the Dockerfile in the main branch to build an image and ran it in a Singularity container. As you know, the Dockerfile uses the requirements.txt file, which contains "transformers==4.57.6". The tests for ESM2, ESMpluspus, and E1 all passed. The attention backend I used was "auto", which defaulted to 'kernel_flash' (I really don't think that the significant drop in performance in the case of my classification task came from the numerical errors/approximations due to flash attention or something. The native ESMC was also run using flash attention)

lhallee:
Thanks for the info @avivko .

Could you share the code you are running with me? You can obscure or remove sensitive details.

Additionally, if you don't mind trying with backend='sdpa' that would rule out flash or flex attention bugs.

avivko:
The code I'm running uses a [fork of this repo](https://github.com/avivko/FastPLMs), and I made a [wrapper](https://github.com/avivko/FastPLMs/blob/main/drorlab_fastplms/embed.py) to run the embedding jobs on our cluster. 

The one thing I figured out is that for the native EMSC, I actually ended up taking the hidden representations from the layer before the last one as the embeddings, and not the last layer (which is the one returned by FastPLMs). That could potentially account for the difference, even though the difference was not that large when I benchmarked it on one of the data folds. It would actually be quite nice if you made it possible to also be able to get the embeddings per layer via the FastPLMs API (should be its own PR, of course).

<img width="1759" height="870" alt="Image" src="https://github.com/user-attachments/assets/92d0d341-7278-41a2-8c9a-6e7558e88e9a" />