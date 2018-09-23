# Machine Learning Engineer Nanodegree
## Capstone Proposal
Arun Dakua
Septmeber 23rd, 2018

## Proposal
_(approx. 2-3 pages)_

Promotion of Indian Regional language movies (Movie Recommendation System for Indian movies).
More than 2000 movies are launched every year in India.
Most of them are aware only of Bollywood movies,which comprises of one third of the movies.
There has been a lot of development in Gujarathi,Bengali,Marathi,Telugu,Tamil and Kannada movies,which could be enjoyed by audience who are exposed to only Bollywood movies.
Currently there are no application or website which would suggest people of movies apart from Bollywood.
The motto here is to provide a service which can recommend Indian regional movies.


### Domain Background
_(approx. 1-2 paragraphs)_

The project is inspired by Movielens(http://movielens.org).
MovieLens is a web site that helps people find movies to watch. It has hundreds of thousands of registered users. It conducts online field experiments in MovieLens in the areas of automated content recommendation, recommendation interfaces, tagging-based recommenders and interfaces, member-maintained databases, and intelligent user interface design.
Currently ,an open source database for Indian movies is not available.We are working on it to collect data from IMDB and various production house. 


In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.

### Problem Statement
_(approx. 1 paragraph)_

Movie recommendation system for Indian movies.Currently no recommmendation system has been created for Indian movies.
The solution to use deep learning on data provided by Grouplens(https://grouplens.org/datasets/movielens/) and replicate on India
Movie database after it is created.

Quantifiable : 
Based on ratings provided by user a user profile is created.
Based on tags of the movies relevance between the movies are created.
Using user profile and movies relevance recommendation are made to the user.

Measurable :
We can use the existing movielens service to test the recommendation made by the engine I developed.

Replicable:
The problem can be reproduced and occurs more than once.

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).

### Datasets and Inputs
_(approx. 2-3 paragraphs)_

Dataset is taken from Grouplens(https://grouplens.org/datasets/movielens/).
Structure of files :

==> movies.csv <==
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
3,Grumpier Old Men (1995),Comedy|Romance
4,Waiting to Exhale (1995),Comedy|Drama|Romance

==> ratings.csv <==
userId,movieId,rating,timestamp
1,110,1.0,1425941529
1,147,4.5,1425942435
1,858,5.0,1425941523
1,1221,5.0,1425941546

==> tags.csv <==
userId,movieId,tag,timestamp
1,318,narrated,1425942391
20,4306,Dreamworks,1459855607
20,89302,England,1400778834
20,89302,espionage,1400778836

==> genome-tags.csv <==
tagId,tag
1,007
2,007 (series)
3,18th century
4,1920s

==> genome-scores.csv <==
movieId,tagId,relevance
1,1,0.024749999999999994
1,2,0.024749999999999994
1,3,0.04899999999999999
1,4,0.07750000000000001


In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

### Solution Statement
_(approx. 1 paragraph)_

Collaberative filtering in Keras.

The idea of using deep learning is similar to that of Matrix Factorization. 
The idea behind matrix factorization is to represent users and items in a lower dimensional latent space . 
(https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))
Matrix factorization is a class of collaborative filtering algorithms used in recommender systems. 
Matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices. This family of methods became widely known during the Netflix prize challenge due to its effectiveness as reported by Simon Funk in his 2006 blog post, where he shared his findings with the research community.

For deep learning implementation, we don’t need them to be matrix form, we want our model to learn the values of embedding matrix itself. The user latent features and movie latent features are looked up from the embedding matrices for specific movie-user combination. These are the input values for further linear and non-linear layers. We can pass this input to multiple relu, linear or sigmoid layers and learn the corresponding weights by any optimization algorithm (Adam, SGD, etc.).

In this section, clearly describe a solution to the problem. The solution should be applicable to project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

To evaluate accuracy of predicted ratings I will use Root Mean Squared Error (RMSE). 
(https://en.wikipedia.org/wiki/Root-mean-square_deviation)
The root-mean-squared error (RMSE)  is a frequently used measure of the differences between values (sample or population values) predicted by a model or an estimator and the values observed. The RMSD represents the square root of the second sample moment of the differences between predicted values and observed values or the quadratic mean of these differences. These deviations are called residuals when the calculations are performed over the data sample that was used for estimation and are called errors (or prediction errors) when computed out-of-sample. The RMSD serves to aggregate the magnitudes of the errors in predictions for various times into a single measure of predictive power. RMSD is a measure of accuracy, to compare forecasting errors of different models for a particular dataset and not between datasets, as it is scale-dependent. 

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

The main components of my neural network:

1.A left neural network layer that creates Users matrix.
A right neural network layer that creates Movies matrix.
The input to the left layer are user and rating data 
The input to the right layer are movies and tags data
A merge layer that takes the dot product of these two vectors to return the predicted rating.

2.This code is based on the approach outlined in Alkahest’s blog post Collaborative Filtering in Keras.

3.Compile the model using Mean Squared Error (MSE) as the loss function and the AdaMax learning algorithm.

4.Split the training and test data in 80/20.

5.Train the model on different epochs. 
Callbacks monitor the validation loss
Save the model weights each time the validation loss has improved 

6.The next step is to actually predict the ratings a random user will give to a random movie. 


In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
