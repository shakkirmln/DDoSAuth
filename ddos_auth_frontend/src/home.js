import React, { Component } from "react";

export class Home extends Component {
  render() {
    let home = (
      <div>
        <div className="limiter">
          <div className="container-login100">
            <div className="login100-form-title ">
              Welcome to our project - Request Count: {this.props.reqCnt}
            </div>
            <div className="login100-form-title ">
              Enhanced Layer of Protection against DDoS
            </div>
            <div>
              One of the major cybercrime is DDoS attack. DDoS attack is a type
              of cyber attack in which the perpetrator uses multiple compromised
              systems to flood the target system with traffic. The goal of a
              DDoS attack is to make the target system unavailable to its
              intended users. In this project, we are going to use 2FA to
              prevent DDoS attack. 2FA is a method of confirming a user's
              claimed identity by utilizing a combination of two different
              components. The first component is face recognition. The second
              component is reading the floating text. The two components are
              independent of each other, so that the attackers cannot automate
              face recognition which requires human and cannot automate reading
              the floating text which the attackers program can't capture.
              again.
            </div>
          </div>
        </div>
        <div id="dropDownSelect1"></div>
      </div>
    );

    return <div>{home}</div>;
  }
}

export default Home;
