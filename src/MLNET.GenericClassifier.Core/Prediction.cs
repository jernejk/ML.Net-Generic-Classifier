using Microsoft.ML.Data;

namespace MLNET.GenericClassifier.Core;

public class Prediction
{
    [ColumnName("PredictedLabel")]
    public string Label { get; set; }

    public float[] Score { get; set; }
}
