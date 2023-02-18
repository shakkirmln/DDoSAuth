import React, { Component } from "react";
import "./css/main.css";
import "./css/util.css";
import "./register.css";
// import Sketch from "react-p5";
import axios from "axios";
// let video;
export class Register extends Component {
  constructor(props) {
    super(props);
    this.state = {
      signup: true,
    };
  }
  // setup(p5, canvasParentRef) {
  //   p5.noCanvas();
  //   video = p5.createCapture(p5.VIDEO);
  //   const v = document.querySelector("video");
  //   let st = "top: 300px;";
  //   v.setAttribute("style", st);
  // }

  // stop() {
  //   const tracks = document.querySelector("video").srcObject.getTracks();
  //   tracks.forEach(function (track) {
  //     track.stop();
  //   });
  // }

  setup2 = async () => {
    const mood = document.getElementById("mood").value;
    if (!mood) {
      alert("Please enter the username!");
      return;
    }
    // video.loadPixels();
    // const image64 = video.canvas.toDataURL();
    const response = await axios.post("http://localhost:5000/register", {
      // image64: image64,
      username: mood,
    });
    if (response.data.status === 200) {
      // this.stop();
      alert("Registered!");
    } else {
      if (response.data.msg) {
        alert(response.data.msg);
      } else {
        alert("Error!");
      }
    }
    this.props.backhome();
  };
  logout = async () => {
    // this.stop();
    const response = await axios.get("http://localhost:5000/close_video_feed");
    console.log(response.data);
    this.props.backhome();
  };

  render() {
    let signup = (
      <div>
        <div className="limiter">
          <div className="container-login100">
            <div className="wrap-login100 p-l-110 p-r-110 p-t-10 p-b-33">
              <span className="login100-form-title m-l-20 p-b-10">
                Sign Up With
              </span>
              <div className="p-t-31 p-b-9">
                <span className="txt1">Username</span>
              </div>
              <div
                className="wrap-input100 validate-input m-b-16"
                data-validate="Username is required"
              >
                <input
                  id="mood"
                  className="input100"
                  type="text"
                  name="username"
                />
                <span className="focus-input100"></span>
              </div>
              {/* <br />
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
              <br />
              <br /> */}

              {this.state.signup ? (
                <img
                  className="live_video"
                  src="http://localhost:5000/video_feed"
                  alt="Sorry, error in the video feed"
                />
              ) : (
                ""
              )}

              <div className="container-login100-form-btn m-t-17">
                <button
                  id="submit"
                  onClick={this.setup2.bind(this)}
                  className="login100-form-btn"
                >
                  Sign Up
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
    return <div>{signup}</div>;
  }
}
export default Register;
