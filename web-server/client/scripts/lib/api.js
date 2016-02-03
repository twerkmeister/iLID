import FetchUtils from "./fetchUtils";
import AudioActions from "../actions/audioActions.js";

const BASE_URL = "http://localhost:9000"; 

const API = {

  postAudio: function(content) {
    const url = `${BASE_URL}/api/upload`;
    const options = {
      method: "POST",
      body: content,
      type: "formdata"
    };

    return FetchUtils.fetchJson(url, options)
      .then(
        (data) => AudioActions.receivePrediction(data),
        (error) => AudioActions.receiveUploadError(error)
      );
  },

  getPredictionForExample(exampleId) {
    const url = `${BASE_URL}/api/example/${exampleId}`;

    return FetchUtils.fetchJson(url)
      .then((data) => AudioActions.receivePrediction(data));
  },
};

export default API