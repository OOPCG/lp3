// SPDX-License-Identifier: UNLICENSED

pragma solidity ^0.8.0;

contract StudentData {

    struct Student{
        uint StudentID;
        string name;
        uint age;
        string course;
    }

    Student[] public students;

    event StudentAdd(uint StudentID, string name);
    event Fallbackcalled(address sender, uint value);

    function AddStudents(uint _StudentID, string memory _name, uint _age, string memory _course) public {
        students.push(Student(_StudentID, _name, _age, _course));
        emit StudentAdd(_StudentID, _name);
    }

    function getStudentCount() public view returns(uint) {
        return students.length;
    }

    function getStudent(uint index) public view returns(uint, string memory, uint, string memory) {
        require(index < students.length, "No Data found");
        Student memory s = students[index];
        return (s.StudentID, s.name, s.age, s.course);
    }

    fallback() external payable { 
        emit Fallbackcalled(msg.sender, msg.value);
    }

    receive() external payable { 
        emit Fallbackcalled(msg.sender, msg.value);
    }
}
