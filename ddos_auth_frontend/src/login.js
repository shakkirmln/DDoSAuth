import React, { Component } from "react";
import "./css/main.css";
import "./css/util.css";
import "./login.css";
import { Home } from "./home.js";
import AudioRecorder from "./audiorecorder.js";
// import Sketch from "react-p5";
import axios from "axios";

var randomWords = require("random-words");

// let video;

export class Login extends Component {
  constructor(props) {
    super(props);
    this.state = {
      verify: false,
      idenity: " ",
      captcha: randomWords(),
      captchaVerify: false,
      blinkCount: Math.floor(Math.random() * 10) + 1,
      videoId: Math.random(),
    };
  }

  componentDidMount() {
    this.writeText();
    this.interval = setInterval(() => {
      this.setState({ captcha: randomWords() });
    }, 10000);
  }

  componentDidUpdate(prevProps, prevState) {
    if (prevState.captcha !== this.state.captcha) {
      this.writeText();
    }
  }

  componentWillUnmount() {
    clearInterval(this.interval);
  }

  writeText() {
    const canvas = document.getElementById("tx1");
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.font = "30px Arial";
      ctx.fillStyle = "black";
      ctx.fillText(this.state.captcha, 30, 50);
    }
  }

  // videoDisplay(p5 = "", canvasParentRef = "") {
  //   p5.noCanvas();
  //   video = p5.createCapture(p5.VIDEO);
  // }

  // stop() {
  //   const tracks = document.querySelector("video").srcObject.getTracks();
  //   tracks.forEach(function (track) {
  //     track.stop();
  //   });
  // }

  logout = async () => {
    // this.stop();
    const response = await axios.get("http://localhost:5000/close_video_feed");
    console.log(response.data);
    this.props.backhome();
  };

  blink = async () => {
    // this.stop();
    const response = await axios.get("http://localhost:5000/stop_blink_count");
    console.log(response.data);
  };

  speechVerify = async (audioBlob) => {
    const formData = new FormData();
    formData.append("audio_file", audioBlob, "recording.wav");
    formData.append("captcha", this.state.captcha);
    const response = await axios.post(
      "http://localhost:5000/captcha",
      formData,
      {
        "content-type": "multipart/form-data",
      }
    );
    alert(response.data);
    if (response.data === "Speech Verified!") {
      this.setState({ captchaVerify: true });
    }
  };

  login = async () => {
    if (this.state.captchaVerify === false) {
      alert("Please verify the captcha!");
      return;
    }
    // video.loadPixels();
    // const image64 = video.canvas.toDataURL();
    const response = await axios.post("http://localhost:5000/login", {
      // image64: image64,
      blinkCount: this.state.blinkCount,
    });
    console.log(response.data.identity);
    if (response.data.identity) {
      // this.stop();
      this.setState({
        verify: true,
        idenity: response.data.identity,
      });
    } else {
      // this.stop();
      if (response.data.msg) {
        alert(response.data.msg);
      } else {
        alert("Not a registered user!");
      }
      this.props.backhome();
    }
  };

  render() {
    let verify = (
      <div>
        <div className="limiter">
          <div className="container-login100">
            <div className="wrap-login100 p-l-110 p-r-110 p-t-10 p-b-33">
              <span className="login100-form-title m-l-20 p-b-10">
                Sign In With
              </span>
              <div>
                Blink your eyes exactly {this.state.blinkCount} times to verify
                (close your eyes for 1 seconds to avoid missing the blink)
              </div>
              <div className="btn-box">
                <button className="btn" onClick={this.blink.bind(this)}>
                  Stop Blink Counter
                </button>
              </div>
              {/* <input />
              <br />
              <br />
              <br />
              <br />
              <br />
              <br />
              <br />
              <br />
              <br />
              <br />
              <br />
              <br />
              <br />
              <br /> */}

              {/* <Sketch setup={this.videoDisplay} draw={this.draw} /> */}
              <img
                className="live_video"
                src={`http://localhost:5000/video_feed/${this.state.videoId}`}
                alt="Sorry, error in the video feed"
              />
              <div className="moving-text">
                <canvas id="tx1"></canvas>
              </div>

              <AudioRecorder speechVerify={this.speechVerify} />

              <div className="container-login100-form-btn m-t-17">
                <button
                  id="submit"
                  onClick={this.login.bind(this)}
                  className="login100-form-btn"
                >
                  Sign In
                </button>
              </div>
              <div className="container-login100-form-btn m-t-17">
                <button
                  onClick={this.logout.bind(this)}
                  className="login100-form-btn"
                >
                  Back!
                </button>
              </div>
            </div>
          </div>
        </div>
        <div id="dropDownSelect1"></div>
      </div>
    );

    return <div>{this.state.verify ? <Home /> : verify}</div>;
  }
}
export default Login;
