# Welcome

This sample project shows how to predict the next's day currency rate using machine learning modules.

**To run this sample**

1.  Register an application on [**Fusion**Fabric.cloud Developer Portal](https://developer.fusionfabric.cloud), and include the **Exchange Rates and Currency Conversion** API.
2. Clone the current project and open it in your file explorer.
3. Copy `config.csv.sample` to `config.csv`, open it, and enter `<%YOUR-CLIENT-ID%>`, and `<%YOUR-SECRET-KEY%>` of the application created at the step 1. 

> The `token_endpoint` is provided by the [Discovery Service](https://developer.preprod.fusionfabric.cloud/documentation?workspace=FusionCreator%20Developer%20Portal&board=Home&uri=oauth2-grants.html#discovery-service) of **Fusion**Fabric.cloud Developer Portal.

4. Open a Command Prompt or Terminal and install the machine learning libraries:

```sh
pip install datetime IPython json keras matplotlib numpy pandas pandas-datareader pygal re requests seaborn sklearn tensorflow

```

5. Run the sample with:

```sh
python Forex-Data-Science.py
```

> To learn how to create this sample project from scratch, follow the tutorial from [Developer Portal Documentation](https://developer.fusionfabric.cloud/documentation?workspace=FusionCreator&board=Home&uri=sample-client-deeplearning.html). 

This sample client application is released under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

