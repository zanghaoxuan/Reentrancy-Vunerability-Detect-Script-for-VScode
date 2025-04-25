event TokensTransfer(
    address indexed _from,
    address indexed _to,
    uint256 amount,
    bool isDone
    );

constructor () public {
    rate = 400;
    wallet = 0xeA9cbceD36a092C596e9c18313536D0EEFacff46;
    cap = 400000000000000000000000;
    openingTime = 1534558186;
    closingTime = 1535320800;

    minInvestmentValue = 0.02 ether;
    
    checksOn = true;
    gasAmount = 25000;
  }

    
  function capReached() public view returns (bool) {
    return tokensRaised >= cap;
  }

    
  function changeRate(uint256 newRate) public onlyOwner {
    rate = newRate;
  }

    
  function closeRound() public onlyOwner {
    closingTime = block.timestamp + 1;
  }

    
  function setToken(ERC20 _token) public onlyOwner {
    token = _token;
  }

    
  function setWallet(address _wallet) public onlyOwner {
    wallet = _wallet;
  }

    
  function changeMinInvest(uint256 newMinValue) public onlyOwner {
    rate = newMinValue;
  }

    
  function setChecksOn(bool _checksOn) public onlyOwner {
    checksOn = _checksOn;
  }

    
  function setGasAmount(uint256 _gasAmount) public onlyOwner {
    gasAmount = _gasAmount;
  }

    
  function setCap(uint256 _newCap) public onlyOwner {
    cap = _newCap;
  }

    
  function startNewRound(uint256 _rate, address _wallet, ERC20 _token, uint256 _cap, uint256 _openingTime, uint256 _closingTime) payable public onlyOwner {
    require(!hasOpened());
    rate = _rate;
    wallet = _wallet;
    token = _token;
    cap = _cap;
    openingTime = _openingTime;
    closingTime = _closingTime;
  }

   
  function hasClosed() public view returns (bool) {
     
    return block.timestamp > closingTime;
  }