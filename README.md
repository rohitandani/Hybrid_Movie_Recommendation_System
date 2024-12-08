Explanation of Directory Structure
1) ml-latest-small/: The root directory of the project.
   app.py: The main script that contains the Flask and Dash applications.
   1.1) templates/: Directory containing HTML templates for the web pages.
      index.html: Template for the home page where users can input their data.
      recommendations.html: Template for displaying the recommended movies.
   1.2) ml-latest-small/ml-latest-small/: Directory for data files.
       ratings.csv: The dataset containing movie ratings.
       movies.csv: The dataset containing movie details.
       tags.csv: The dataset containing movie tags.
       requirements.txt: A text file listing the Python packages required for the project. This is useful for setting up the project environment.

******Execution :- git clone https://github.com/rohitandani/Hybrid_Movie_Recommendation_System.git
             
             cd Hybrid_Movie_Recommendation_System/ml-latest-small
             pip install -r requirements.txt****
**
In the Thesis you will see the detailed of the code and output of vaiour Heatmap and Graph whosing the impact on the input on data after applying various algorithms. Also , you can get the recommendation of the movie based on text as output via executing the **ml-latest-small/ml-latest-small/Hybrid Movie Recommendation System.ipynb**.
