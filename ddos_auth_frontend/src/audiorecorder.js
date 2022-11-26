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
  }

  controlAudio(status) {
    this.setState({
      status,
    });
  }

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
      startCallback: (e) => {
        console.log("succ start", e);
      },
      pauseCallback: (e) => {
        console.log("succ pause", e);
      },
      stopCallback: async (e) => {
        this.setState({
          audioSrc: window.URL.createObjectURL(e),
        });
        console.log("succ stop", e);
        const formData = new FormData();
        formData.append("audio_file", e, "recording.wav");
        await axios.post("http://localhost:5000/captcha", formData, {
          "content-type": "multipart/form-data",
        });
      },
      onRecordCallback: (e) => {
        console.log("recording", e);
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
              Start
            </button>
            <button className="btn" onClick={() => this.controlAudio("paused")}>
              Pause
            </button>
            <button
              className="btn"
              onClick={() => this.controlAudio("inactive")}
            >
              Stop
            </button>
          </div>
        </AudioAnalyser>
      </div>
    );
  }
}
