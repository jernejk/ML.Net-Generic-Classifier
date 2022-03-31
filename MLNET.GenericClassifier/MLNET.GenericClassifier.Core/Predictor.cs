using Microsoft.Extensions.ML;
using Microsoft.ML;

namespace MLNET.GenericClassifier.Core;

public interface IPredictor
{
    void LoadFromStream(Stream stream);
    void LoadFromFilePath(string modelPath);
    void SetModel(ITransformer mlModel);

    List<string> GetCategories();
}

public interface IPredictor<TInput> : IPredictor
    where TInput : class
{
    void SetPredictionPool(PredictionEnginePool<TInput, Prediction> predictionPool);

    Prediction Predict(TInput input);
    Dictionary<string, float> PredictionList(TInput input);
}

public class Predictor<TInput> : IPredictor<TInput>
    where TInput : class
{
    private readonly MLContext _mlContext;

    private ITransformer? _mlModel;
    private List<string>? _categories;
    private PredictionEnginePool<TInput, Prediction>? _predictionEnginePool;

    public Predictor()
    {
        _mlContext = new MLContext();
    }

    /// <summary>
    /// Used for Blazor WASM and Azure Functions where we don't have direct access to the model.
    /// </summary>
    public void LoadFromStream(Stream stream)
    {
        // Load model from file.
        SetModel(_mlContext.Model.Load(stream, out var _));
    }

    /// <summary>
    /// Used for Console applications where we can access the file directly.
    /// </summary>
    public void LoadFromFilePath(string modelPath)
    {
        // Load model from file.
        using var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        SetModel(_mlContext.Model.Load(stream, out var _));
    }

    /// <summary>
    /// Use this when using ASP.NET Core Dependency Injection.
    /// </summary>
    public void SetPredictionPool(PredictionEnginePool<TInput, Prediction> predictionPool)
    {
        _predictionEnginePool = predictionPool;
    }

    public void SetModel(ITransformer mlModel)
    {
        // Reset category cache.
        _categories = null;
        _mlModel = mlModel;
    }

    public Prediction Predict(TInput input)
    {
        if (_predictionEnginePool != null)
        {
            // Used for scalable applications, recycling prediction engines.
            return _predictionEnginePool.Predict(input);
        }

        // Used for console applications, MAUI and Blazor where scalability might not be a problem.
        PredictionEngine<TInput, Prediction>? predictionEngine = _mlContext.Model.CreatePredictionEngine<TInput, Prediction>(_mlModel);
        return predictionEngine.Predict(input);
    }

    public Dictionary<string, float> PredictionList(TInput input)
    {
        Prediction prediction = Predict(input);
        if (prediction == null)
        {
            return new Dictionary<string, float>();
        }

        // Get and cache categories.
        List<string> categories = GetCategories();
        return MicrosoftMLExtensions.GetCategoriesWithScore(categories, prediction.Score);
    }

    public List<string> GetCategories()
    {
        if (_categories != null)
        {
            return _categories;
        }

        DataViewSchema schema = GetOutputSchema();
        _categories = schema.GetCategories();

        return _categories;
    }

    public DataViewSchema GetOutputSchema()
    {
        PredictionEngine<TInput, Prediction> predEngine = _predictionEnginePool != null
            ? _predictionEnginePool.GetPredictionEngine()
            : _mlContext.Model.CreatePredictionEngine<TInput, Prediction>(_mlModel);

        return predEngine.OutputSchema;
    }
}
