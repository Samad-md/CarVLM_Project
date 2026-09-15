import { useState } from "react";
import { predictVehicle } from "./services/api";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];

    setImage(file);
    setResult(null);

    if (file) {
      setPreview(URL.createObjectURL(file));
    }
  };

  const handlePredict = async () => {
    if (!image) {
      alert("Please select vehicle image");
      return;
    }

    try {
      setLoading(true);

      const response = await predictVehicle(image);

      console.log("Backend Response:", response);

      setResult(response);
    } catch (error) {
      console.log(error);

      setResult({
        predicted_vehicle: "Prediction Failed",
        status: "API Error",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>🚗 CarVLM Vehicle Recognition</h1>

      <div className="upload-box">
        <input type="file" accept="image/*" onChange={handleImageChange} />
      </div>

      {preview && (
        <div className="image-section">
          <h2>Selected Vehicle</h2>

          <img src={preview} alt="vehicle" className="preview" />
        </div>
      )}

      <button
        className="predict-btn"
        onClick={handlePredict}
        disabled={loading}
      >
        {loading ? "Analyzing Vehicle..." : "Predict Vehicle"}
      </button>

      <div className="result-box">
        <h2>Prediction Result</h2>

        {loading && <h3>🔍 AI is analyzing image...</h3>}

        {result && !loading && (
          <div className="vehicle-card">
            <h1>🚘 {result.predicted_vehicle}</h1>

            {result.specifications && (
              <div className="details">
                <p>
                  <b>Brand:</b> {result.specifications.make}
                </p>

                <p>
                  <b>Model:</b> {result.specifications.model}
                </p>

                <p>
                  <b>Year:</b> {result.specifications.year}
                </p>

                <p>
                  <b>Vehicle Type:</b> {result.specifications.vehicle_type}
                </p>
              </div>
            )}

            <p className="status">✅ {result.status}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
