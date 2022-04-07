using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace MLNET.GenericClassifier.Trainer.Core;

public interface IBaseTrainer
{
    public MLContext MlContext { get; }

    RunDetail<MulticlassClassificationMetrics> AutoTrain(IDataView? trainingData, uint maxTimeInSec, IDataView? validationdata = null);
    void SaveModel(string modelSavePath, ITransformer model);
    void SaveModel(Stream stream, ITransformer model);
}

public abstract class BaseTrainer : IBaseTrainer
{
    protected IDataView? _trainingDataView;

    protected BaseTrainer(MLContext? mlContext = null)
    {
        MlContext = mlContext ?? new MLContext();
    }

    public MLContext MlContext { get; }

    public RunDetail<MulticlassClassificationMetrics> AutoTrain(IDataView trainingData, uint maxTimeInSec, IDataView? validationdata = null)
    {
        _trainingDataView = trainingData;

        MulticlassExperimentSettings experimentSettings = new()
        {
            MaxExperimentTimeInSeconds = maxTimeInSec,
            OptimizingMetric = MulticlassClassificationMetric.MacroAccuracy
        };

        ColumnInformation columnInfo = GetColumnInformation();
        ConfigureExperiment(experimentSettings, columnInfo);

        MulticlassClassificationExperiment experiment = MlContext.Auto()
            .CreateMulticlassClassificationExperiment(experimentSettings);


        ExperimentResult<MulticlassClassificationMetrics>? result =
            validationdata != null
            ? experiment.Execute(_trainingDataView,  validationdata, columnInfo)
            : experiment.Execute(_trainingDataView, columnInfo);

        return result.BestRun;
    }

    public void SaveModel(string modelSavePath, ITransformer model)
    {
        // Save training model to disk.
        MlContext.Model.Save(model, _trainingDataView!.Schema, modelSavePath);
    }

    public void SaveModel(Stream stream, ITransformer model)
    {
        // Save training model to disk.
        MlContext.Model.Save(model, _trainingDataView!.Schema, stream);
    }

    /// <summary>
    /// Column definition is usually one of the few things are different between various classifications
    /// as number of input columns can be different.
    /// 
    /// Also, some ML model benefits certain columns to be treated as hashes opposed to text featurize or we need to skip a couple of columns.
    /// </summary>
    /// <returns>Returns column information.</returns>
    protected abstract ColumnInformation GetColumnInformation();

    /// <summary>
    /// Add additional configuration for the experiment.
    /// </summary>
    /// <param name="experimentSetting">Pre-build multi-class experiment settings</param>
    /// <param name="columnInfo">Column info loaded from calling GetColumnInformation()</param>
    protected virtual void ConfigureExperiment(MulticlassExperimentSettings experimentSetting, ColumnInformation columnInfo) { }
}
