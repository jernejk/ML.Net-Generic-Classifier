namespace MLNET.GenericClassifier;

/// <summary>
/// Class representing the training data.
/// Can also be used for prediction or we can create a separate class without the predicted label property.
/// </summary>
public class GitComment
{
    /// <summary>
    /// This is the column we want to predict.
    /// We only needs this property for training and we can have a different class without the property that represents the label.
    /// </summary>
    [LoadColumn(0)]
    [ColumnName("col0")]
    public string? Emoji { get; set; }

    [LoadColumn(1)]
    [ColumnName("col1")]
    public string? CommitMessage { get; set; }

    public override string ToString() => CommitMessage!;
}

/// <summary>
/// Generic trainer only needs to know the definition of columns.
/// </summary>
public class GitmoTrainer : BaseTrainer
{
    public GitmoTrainer(MLContext? mlContext = null)
        : base(mlContext) { }

    protected override ColumnInformation GetColumnInformation()
    {
        ColumnInformation columnInfo = new() { LabelColumnName = "col0" };
        columnInfo.TextColumnNames.Add("col1");

        return columnInfo;
    }

    protected override void ConfigureExperiment(MulticlassExperimentSettings experimentSetting, ColumnInformation columnInfo)
    {
        base.ConfigureExperiment(experimentSetting, columnInfo);

        // We can decide to remove some of the trainers that we think will work terribly or takes to long to train.
        experimentSetting.Trainers.Remove(MulticlassClassificationTrainer.SymbolicSgdLogisticRegressionOva);
    }
}
