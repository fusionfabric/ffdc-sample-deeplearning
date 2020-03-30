# Welcome

This sample project shows how to predict the next's day currency rate using machine learning modules.

**To run this sample**

> You need a recent installation of [Python](https://www.python.org/) and [Jupyter](https://jupyter.org/). However, due to the dependency on the **keras** library, you must install a compatible Python version, such as 2.7 or 3.6, or previous versions. See the library repository [Readme.md](https://github.com/keras-team/keras/blob/master/README.md) for details.
> If you are new to Python, check out the [Beginner's Guide to Python](https://wiki.python.org/moin/BeginnersGuide), which contains instructions on how to install Python.

1.  Register an application on [**Fusion**Fabric.cloud Developer Portal](https://developer.fusionfabric.cloud), and include the [Exchange Rates and Currency Conversion](https://developer.fusionfabric.cloud/api/fxrate-v1-f1ee44fa-bdd1-44ed-b4b5-50298b82f0d/docs) API.
2. Clone the current project and open it in your file explorer.
3. Copy `config.csv.sample` to `config.csv`, open it, and enter `<%YOUR-CLIENT-ID%>`, and `<%YOUR-SECRET-KEY%>` of the application created at the step 1. 

> The `token_endpoint` is provided by the [Discovery Service](https://developer.fusionfabric.cloud/documentation/oauth2-grants#discovery-service) of **Fusion**Fabric.cloud Developer Portal.

4. Open a Command Prompt or Terminal and install the machine learning libraries:

```sh
pip install datetime IPython json keras matplotlib numpy pandas pandas-datareader pygal re requests seaborn sklearn tensorflow

```

5. Run the sample with:

```sh
python Forex-Data-Science.py
```

> To learn how to create this sample project from scratch, follow the tutorial from [Developer Portal Documentation](https://developer.fusionfabric.cloud/documentation/sample-client-deeplearning). 

This sample client application is released under the MIT License. See [LICENSE](LICENSE) for details.

