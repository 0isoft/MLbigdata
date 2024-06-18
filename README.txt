This project explores various autoencoder architectures for matrix completion on the MovieLens and Netflix datasets. The primary goal is to predict missing ratings in user-item matrices using autoencoders, including both simple and complex designs. This README provides instructions for running the code and evaluating the pre-trained models.

-----
DIRECTORY STRUCTURE

--> movielens_v2.ipynb: Implementation of AutoRec on the MovieLens dataset.

--> netflix_v4.ipynb: Implementation of AutoRec on the Netflix dataset.

--> evaluate_netflix.ipynb: Notebook for evaluating the AutoRec model on the Netflix dataset.

--> netflix_v5.ipynb: Implementation of a deeper autoencoder on the Netflix dataset.

--> movielens_v4.ipynb: Attempt at implementing a VAE autoencoder on the MovieLens dataset.



The directory also contains pretrained model files:

--> movielens_simple_autoencoder.pth
--> movielens_simple_autoencoder_best_param.pth
--> netflix_simple_autoencoder.pth
--> netflix_complex_autoencoder.pth

----

RUNNING THE CODE
1. Simple AutoRec Model on MovieLens dataset:
	-Open movielens_v2.ipynb.
	-Run preprocessing cells (Cells 1 through 8).
	-Run model definition and cost function cells (Cells 9 and 10).
	-To load the pre-trained model (movielens_simple_autoencoder.pth) and avoid training from scratch:
	Uncomment the line in Cell 15: #model = loaded_model # <------- enable this to avoid having to train the model.
	Run the remaining cells to evaluate the model.

2. AutoRec with Adjusted Hyperparameters:
	In movielens_v2.ipynb, to load the adjusted model (movielens_simple_autoencoder_best_param.pth):
	Uncomment the line in Cell 24: #model = loaded_model # <------- enable this to avoid having to train the model.
	Run the remaining cells to evaluate the model.

3. Simple AutoRec Model on Netflix dataset:
	- Open netflix_v4.ipynb and ensure the model is saved as netflix_simple_autoencoder.pth.
	- To evaluate this model, open evaluate_netflix.ipynb:
	This notebook will load, normalize the data, and proceed with the evaluation based on the simple AutoRec architecture.
	- Run all cells as they are to load the pre-trained model and perform evaluation.

4. Deeper Autoencoder:
	- The deeper autoencoder is trained in netflix_v5.ipynb and saved as netflix_complex_autoencoder.pth.
	-To evaluate this model, use evaluate_netflix.ipynb:
	The notebook can load and evaluate the deeper autoencoder architecture.
	Run all cells to load the pre-trained model and perform evaluation.

5. VAE Autoencoder: The implementation attempt for a variational autoencoder is in movielens_v4.ipynb.
This notebook provides suboptimal results and has been left unfinished.

Other notebooks are included for completeness. For instance, movielens_v2 includes attempts
at a deep autoencoder with regularization and dropout layers. These features were dropped in subsequent iterations.

---
Dependencies:
->PyTorch
->scikit-learn
->Pandas
->NumPy
->Matplotlib