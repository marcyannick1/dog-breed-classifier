import React, { useState } from 'react';

export default function Form() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [url, setUrl] = useState('');
  const [result, setResult] = useState(null);
  const [uploadMethod, setUploadMethod] = useState('file');

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleUrlChange = (e) => {
    const url = e.target.value;
    setUrl(url);
    setPreview(url);
  };

  const handlePredict = async () => {
    if (!image && !url) return;

    const formData = new FormData();
    if (image) formData.append('file', image);
    if (url) formData.append('url', url);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Échec de la prédiction : ', error);
      setResult({ error: 'Échec de la prédiction' });
    }
  };

  return (
    <div className="max-w-xl mx-auto mt-4 p-6 bg-white rounded-2xl shadow-md space-y-4 border-2 border-blue-800">
      <h2 className="text-xl font-semibold text-gray-800">Uploader une image de chien</h2>

      <div className="flex space-x-4">
        <button
          className={`w-full py-2 px-4 rounded-xl ${uploadMethod === 'file' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-800'}`}
          onClick={() => setUploadMethod('file')}
        >
          Image depuis l'ordinateur
        </button>
        <button
          className={`w-full py-2 px-4 rounded-xl ${uploadMethod === 'url' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-800'}`}
          onClick={() => setUploadMethod('url')}
        >
          Image via URL
        </button>
      </div>

      {uploadMethod === 'file' && (
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4
            file:rounded-full file:border-0 file:text-sm file:font-semibold
            file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />
      )}

      {uploadMethod === 'url' && (
        <div>
          <input
            type="text"
            placeholder="Entrez l'URL de l'image"
            value={url}
            onChange={handleUrlChange}
            className="block w-full text-sm text-gray-500 border-gray-600 p-2 rounded-md border"
          />
        </div>
      )}

      {preview && (
        <div className="mt-4">
          <p className="text-gray-700 font-medium">Aperçu :</p>
          <img src={preview} alt="preview" className="w-full h-80 rounded-lg shadow" />
        </div>
      )}

      {/* Bouton de prédiction */}
      <button
        onClick={handlePredict}
        className="w-full bg-blue-600 text-white py-2 px-4 rounded-xl hover:bg-blue-700"
      >
        Prédire la race
      </button>

      {/* Affichage des résultats */}
      {result && (
        <div className="text-green-600 font-bold mt-4">
          {result.error ? result.error : (
            <div>
              <p className="underline">Prédictions :</p>
              <ul className="list-disc text-left mt-2">
                {result.top_predictions.map((prediction, index) => (
                  <li key={index}>
                    {prediction.breed.split('-')[1]} : {prediction.confidence.toFixed(2)}%
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
