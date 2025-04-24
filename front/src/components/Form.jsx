import React, { useState } from 'react';

export default function Form() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handlePredict = async () => {
    if (!image) return;

    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Prediction failed:', error);
      setResult({ error: 'Échec de la prédiction' });
    }
  };

  return (
    <div className="max-w-xl mx-auto mt-4 p-6 bg-white rounded-2xl shadow-md space-y-4 border-2 border-blue-800">
      <h2 className="text-xl font-semibold text-gray-800">Uploader une image de chien</h2>

      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4
          file:rounded-full file:border-0 file:text-sm file:font-semibold
          file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
      />

      {preview && (
        <div className="mt-4">
          <p className="text-gray-700 font-medium">Aperçu :</p>
          <img src={preview} alt="preview" className="w-full h-80 rounded-lg shadow" />
        </div>
      )}

      <button
        onClick={handlePredict}
        className="w-full bg-blue-600 text-white py-2 px-4 rounded-xl hover:bg-blue-700"
      >
        Prédire la race
      </button>

      {result && (
        <div className="text-center text-green-600 font-bold mt-4">
          {result.error ? result.error : result.labels?.join(', ')}
        </div>
      )}
    </div>
  );
}
