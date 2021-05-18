# Capstone project: The Gait Analyzer App
Authors: Ananya Swaminathan, Yvette Tan

This app was created as a tool to analyze gait data from patients with ALS, Parkinson's, Huntington's, or control patients, and to predict the neurological disease condition based on the data. The data used in this app are stride to stride measures of footfall contact times, derived from readings by force-sensitive resistors that give outputs proportional to the force under each foot. Each patient's data includes time series data of 13 signals (6 each for  the left and right sides): 

1. Elapsed time (sec)
2. Left stride interval (sec)
3. Right stride interval (sec)
4. Left swing interval (sec)
5. Right swing interval (sec)
6. Left swing interval (% of stride)
7. Right swing interval (% of stride)
8. Left stance interval (sec)
9. Right stance interval (sec)
10. Left stance interval (% of stride)
11. Right stance interval (% of stride)
12. Double support interval (sec)
13. Double support interval (% of stride)

Users can select either one of the provided text files in the gait_data directory, or can choose their own data. This app graphs the raw and processed data of the chosen waveform signal, and provides a prediction for the disease state of the patient. A Random Forest model was trained to make these predictions, with a test set accuracy of 0.9375. 

****NOTE:
Our model is by no means a method of diagnosis and HAS NOT been scientifically validated. This model has been trained and tested on a small dataset (48 and 16 respectively), and therefore may not work accurately or effectively in a real-world setting. Please consult a professional if you have any concerns related to your gait.** 


## Live app and Documentation:
The link to the live app is as follows: https://ananyas713.shinyapps.io/Gait_Analyzer/

Documentation for the app can be found here: https://ananyas713.shinyapps.io/Gait_Analyzer/#section-about

## App requirements:
While this shiny app flexdashboard was created in R, the data processing and modeling were done through Python. 
- This app was created using Python 3.7 and R 3.6. 
- Required R libraries include: shiny, reticulate, tidyverse, and plotly. 
- Required Python libraries include: numpy, pandas, scikit-learn, and glob2. 

If running the app locally, it is highly recommended to use Anaconda and create a virtual environment with these dependencies. 

The app runs through the final_Project.Rmd file. Other files that are required in the same repository as the Rmd include the *get_model_stats.py* and *get_model_pred.py* Python files, which run the prediction models, as well as the *model.pkl* and *modelstats.npy* files, which hold a snapshot of the trained model and information about its performance, respectively. The data used was from the Gait in Neurodegerative Disease Database on the PhysioNet database; more information can be found at this link: https://www.physionet.org/content/gaitndd/1.0.0/

### Contributions
- Ananya: outlier removal, mean imputation, statistical feature generation; PCA, model building; Shiny app user interface, raw data output, disease prediction output, and documentation; Shiny app deployment (virtualenv and .Rprofile creation)
- Yvette: datafile input, data concatenation, patient numbering, time windowing, statistical feature generation; grid search, model building; Shiny app processed data output; README documentation


## References: 
Hausdorff JM, Lertratanakul A, Cudkowicz ME, Peterson AL, Kaliton D, Goldberger AL. Dynamic markers of altered gait rhythm in amyotrophic lateral sclerosis. J Applied Physiology; 88:2045-2053, 2000.
