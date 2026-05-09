import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setImageUrl(URL.createObjectURL(event.target.files[0]));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('/detect', formData, {
        responseType: 'blob',
      });
      const url = URL.createObjectURL(response.data);
      setImageUrl(url);
    } catch (error) {
      console.error('Error uploading file:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Object Detection App</h1>
        <p>Upload an image to detect objects</p>
      </header>
      <main>
        <form onSubmit={handleSubmit}>
          <input type="file" accept="image/*" onChange={handleFileChange} />
          <button type="submit" disabled={!selectedFile || loading}>
            {loading ? 'Detecting...' : 'Detect Objects'}
          </button>
        </form>
        {imageUrl && (
          <div className="image-container">
            <img src={imageUrl} alt="Detected objects" />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;