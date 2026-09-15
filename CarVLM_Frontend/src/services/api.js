import axios from "axios";

export async function predictVehicle(image) {
  const formData = new FormData();
  formData.append("file", image);

  const response = await axios.post("http://127.0.0.1:8000/predict", formData);

  console.log(response.data);

  return response.data;
}
