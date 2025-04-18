import axios from "axios";

export const predictAQI = async (data) => {
  return axios.post('http://localhost:8000/predict_aqi/', data, {
    headers: {
      'Content-Type': 'application/json',
    },
    transformRequest: [(data) => {
      // Ensure numeric values
      const processed = {};
      for (const key in data) {
        processed[key] = typeof data[key] === 'string' ? 
          parseFloat(data[key]) || 0 : 
          data[key];
      }
      return JSON.stringify(processed);
    }]
  });
};