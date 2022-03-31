namespace MLNET.GenericClassifier;

public class TransactionData
{
    [LoadColumn(0)]
    [ColumnName("Transaction Name")]
    public string TransactionName { get; set; }

    [LoadColumn(1)]
    [ColumnName("Category")]
    public string Category { get; set; }
}

public class TransactionTrainer : BaseTrainer
{
    public TransactionTrainer(MLContext? mlContext = null)
        : base(mlContext) { }

    protected override ColumnInformation GetColumnInformation()
    {
        ColumnInformation columnInfo = new() { LabelColumnName = "Category" };
        columnInfo.TextColumnNames.Add("Transaction Name");

        return columnInfo;
    }
}
