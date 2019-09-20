# Python Model Predictions (Churn) using SSE for Qlik
### Leveraging local Random Forest, SVC, and KNN models on disk.
#### *Example apps for Qlik Sense & QlikView*

![Sheet 1](../assets/assets/churn-sheet-1.png?raw=true)


## REQUIREMENTS

- **Assuming prerequisite: [Python with Qlik Sense AAI – Environment Setup](https://docs.google.com/viewer?url=https://github.com/danielpilla/qlik-python-sse-churn-model-predictions/blob/assets/assets/DPI%20-%20Qlik%20Sense%20AAI%20and%20Python%20Environment%20Setup.pdf?raw=true)**
	- This is not mandatory and is intended for those who are not as familiar with Python to setup a virtual environment. Feel free to follow the below instructions flexibly if you have experience.
- Qlik Sense June 2017+
- QlikView November 2017+
    - *This guide is designed for Qlik Sense but may be used with QlikView. See how to setup Analytic Connections within QlikView [here](https://help.qlik.com/en-US/qlikview/November2017/Subsystems/Client/Content/Analytic_connections.htm)*
- Python 3.5.3 64 bit *(3.4+ but tested on 3.5.3)*
- Python Libraries: grpcio, numpy, pandas, sklearn, scipy

## LAYOUT

- [Prepare your Project Directory](#prepare-your-project-directory)
- [Install Python Libraries and Required Software](#install-python-libraries-and-required-software)
- [Setup an AAI Connection in the QMC](#setup-an-aai-connection-in-the-qmc)
- [Copy the Package Contents and Import Examples](#copy-the-package-contents-and-import-examples)
- [Prepare And Start Services](#prepare-and-start-services)
- [Leveraging Existing Models and Making Predictions from Qlik](#leveraging-existing-models-and-making-predictions-from-qlik)
- [Configure your SSE as a Windows Service](#configure-your-sse-as-a-windows-service)

 
## PREPARE YOUR PROJECT DIRECTORY
>### <span style="color:red">ALERT</span>
><span style="color:red">
>Virtual environments are not necessary, but are frequently considered a best practice when handling multiple Python projects.
></span>

1. Open a command prompt
2. Make a new project folder called QlikSenseAAI, where all of our projects will live that leverage the QlikSenseAAI virtual environment that we’ve created. Let’s place it under ‘C:\Users\{Your Username}’. If you have already created this folder in another guide, simply skip this step.

3. We now want to leverage our virtual environment. If you are not already in your environment, enter it by executing:

```shell
$ workon QlikSenseAAI
```

4. Now, ensuring you are in the ‘QlikSenseAAI’ folder that you created (if you have followed another guide, it might redirect you to a prior working directory if you've set a default, execute the following commands to create and navigate into your project’s folder structure:
```
$ cd QlikSenseAAI
$ mkdir Churn
$ cd Churn
```


5. Optionally, you can bind the current working directory as the virtual environment’s default. Execute (Note the period!):
```shell
$ setprojectdir .
```
6. We have now set the stage for our environment. To navigate back into this project in the future, simply execute:
```shell
$ workon QlikSenseAAI
```

This will take you back into the environment with the default directory that we set above. To change the
directory for future projects within the same environment, change your directory to the desired path and reset
the working directory with ‘setprojectdir .’


## INSTALL PYTHON LIBRARIES AND REQUIRED SOFTWARE

1. Open a command prompt or continue in your current command prompt, ensuring that you are currently within the virtual environment—you will see (QlikSenseAAI) preceding the directory if so. If you are not, execute:
```shell
$ workon QlikSenseAAI
```
2. Execute the following commands. If you have followed a previous guide, you have more than likely already installed grpcio):

```shell
$ pip install grpcio
$ python -m pip install grpcio-tools
$ pip install numpy
$ pip install pandas
$ pip install scikit-learn==0.19.1
$ pip install scipy
```

## SET UP AN AAI CONNECTION IN THE QMC

1. Navigate to the QMC and select ‘Analytic connections’
2. Fill in the **Name**, **Host**, and **Port** parameters -- these are mandatory.
    - **Name** is the alias for the analytic connection. For the example qvf to work without modifications, name it 'PythonChurn'
    - **Host** is the location of where the service is running. If you installed this locally, you can use 'localhost'
    - **Port** is the target port in which the service is running. This module is setup to run on 50072, however that can be easily modified by searching for ‘-port’ in the ‘ExtensionService_churn.py’ file and changing the ‘default’ parameter to an available port.
3. Click ‘Apply’, and you’ve now created a new analytics connection.


## COPY THE PACKAGE CONTENTS AND IMPORT EXAMPLES

1. Now we want to setup our directions service and app. Let’s start by copying over the contents of the example
    from this package to the ‘..\QlikSenseAAI\Churn\’ location. Alternatively you can simply clone the repository.
2. After copying over the contents, go ahead and import the example qvf found [here](..assets/assets/Churn%20Predictions.qvf?raw=true) or the example qvw (if using QlikView) [here](../assets/assets/DPI%20-%20Python%20Churn%20Predictions.qvw?raw=true).
3. Lastly, import the *Climber KPI* extension found [here](https://github.com/ClimberAB/ClimberKPI) if you are using Qlik Sense.


## PREPARE AND START SERVICES

1. At this point the setup is complete, and we now need to start the directions extension service. To do so, navigate back to the command prompt. Please make sure that you are inside of the virtual environment.
2. Once at the command prompt and within your environment, execute (note two underscores on each side):
```shell
$ python ExtensionService_churn.py
```
3. We now need to restart the Qlik Sense engine service so that it can register the new SSE service. To do so,
    navigate to windows Services and restart the ‘Qlik Sense Engine Service’
4. You should now see in the command prompt that the Qlik Sense Engine has registered the function *ModelPredict()* from the extension service over port 50072, or whichever port you’ve chosen to leverage.


## LEVERAGING EXISTING MODELS AND MAKING PREDICTIONS FROM QLIK

1. About this package:
	- This package demonstrates the ability to make predictions against existing models that have already been created. These models could live at a REST endpoint potentially, or like this example, live on disk (using something like pickle). We can not only leverage these models on demand, but we can also leverage all of them simultaneously to compare model accuracies. This approach allows the team who owns the models to own them outright and continue to tweak and modify them without any affect to Qlik Sense (spare any changes to what fields they might take).

2. The function:
	- This extension service contains a function called *ModelPredict()* which takes three arguments:
		- *ModelName (string)*:
			- SVC
			- KNN
			- RF
		- *Concatenated list of column names*
		- *Concatenated list of field (record) values*
	- The three models it supports are SVC (Support Vector Machines), KNN (k-Nearest-Neighbors), and RF (Random Forest) -- these models (as well as the scaler) were created from the historical dataset, excluding a small chunk that we are using for our predictions inside of the Qlik app.

3. The app:
	- On the dashboard, you are required to select a single Model, at which poin tthe data is sent to Python, and the predictions are returned. These returned predictions are then used in many visualizations to calculate statistical accuracy.
	- On the Challenger Models sheet, overall scores from all models are calculated. Here, you are able to select dimensions from the panes on the left hand side to see how accuracy varies by dimension by model. 

![Sheet 1](../assets/assets/churn-sheet-1.png?raw=true)

![Sheet 2](../assets/assets/churn-sheet-2.png?raw=true)

![QV Sheet 1](../assets/assets/QVChurn.png?raw=true)

![QV Sheet 2](../assets/assets/QVChurn2.png?raw=true)
 
## CONFIGURE YOUR SSE AS A WINDOWS SERVICE

Using NSSM is my personal favorite way to turn a Python SSE into a Windows Service. You will want to run your SSEs as services so that they startup automatically and run in the background.
1. The **Path** needs to be the location of your desired Python executable. If you've followed my guide and are using a virtual environment, you can find that under 'C:\Users\\{USERNAME}\Envs\QlikSenseAAI\Scripts\python.exe'.
2. the **Startup directory** needs to be the parent folder of the extension service. Depending on what guide you are following, the folder needs to contain the '_\_main\_\_.py' file or the 
'ExtensionService_{yourservicename).py' file.
3. The **Arguments** parameter is then just the name of the file that you want Python to run. Again, depending on the guide, that will either be the '\_\_main\_\_.py' file or the 'ExtensionService_{yourservicename).py' file.

**Example:**

![ServiceExample](../assets/assets/PythonAsAService.png?raw=true)
