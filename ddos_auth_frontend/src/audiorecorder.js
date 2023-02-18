import React, { Component } from "react";
import axios from "axios";
import AudioAnalyser from "react-audio-analyser";
import "./audiorecorder.css";

export default class AudioRecorder extends Component {
  constructor(props) {
    super(props);
    this.state = {
      status: "",
    };
    this.speechVerify = props.speechVerify;
  }

  controlAudio = async (status) => {
    this.setState({
      status,
    });
    if (status === "recording") {
      const response = await axios.get(
        "http://localhost:5000/start_audio_stream"
      );
      console.log(response.data);
    }
    if (status === "inactive") {
      const response = await axios.get(
        "http://localhost:5000/close_audio_stream"
      );
      console.log(response.data);
    }
  };

  componentDidMount() {
    this.setState({
      audioType: "audio/wav",
    });
  }

  render() {
    const { status, audioSrc, audioType } = this.state;
    const audioProps = {
      audioType,
      status,
      audioSrc,
      timeslice: 1000,
      // startCallback: (e) => {
      //   console.log("succ start", e);
      // },
      // pauseCallback: (e) => {
      //   console.log("succ pause", e);
      // },
      // onRecordCallback: (e) => {
      //   console.log("recording", e);
      // },
      stopCallback: async (e) => {
        this.setState({
          audioSrc: window.URL.createObjectURL(e),
        });

        this.speechVerify(e);
      },
      errorCallback: (err) => {
        console.log("error", err);
      },
    };
    return (
      <div>
        <AudioAnalyser {...audioProps}>
          <div className="btn-box">
            <button
              className="btn"
              onClick={() => this.controlAudio("recording")}
            >
              &#9658;
            </button>
            <button
              className="btn"
              onClick={() => this.controlAudio("inactive")}
            >
              Validate
            </button>
          </div>
        </AudioAnalyser>
      </div>
    );
  }
}
