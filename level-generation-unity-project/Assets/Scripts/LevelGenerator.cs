using System.Collections;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;

// Part of the Sketch based level generation system written in python
// Requires the Newtonsoft JsonDotNet package
// (C) Alexander Hjelm, 2019

public class LevelGenerator : MonoBehaviour
{
    [SerializeField] private GameObject _roomQuadPrefab;

    private List<int[]> rectangles = new List<int[]>();
    private List<int[]> circles = new List<int[]>();

    void Start()
    {
        string mapFilePath = "Assets/Resources/map.json";

        // Read the map json file to a string
        StreamReader reader = new StreamReader(mapFilePath); 
        string jsonStr = reader.ReadToEnd();
        Debug.Log(jsonStr);

        // Convert json string to a dictionary
        Dictionary<string, int[]> shapesDict = JsonConvert.DeserializeObject<Dictionary<string, int[]>>(jsonStr);

        // Parse each element and determine the shape type
        foreach(string key in shapesDict.Keys)
        {
            int[] coords = shapesDict[key];
            string typeStr = key.Split(char.Parse("_"))[0];

            if(string.Equals(typeStr, "rect"))
                rectangles.Add(coords);
            else if(string.Equals(typeStr, "circle"))
                circles.Add(coords);
        }

        // Create the level
        foreach(int[] coords in rectangles)
        {
            int x1 = coords[0];
            int y1 = coords[1];
            int x2 = coords[2];
            int y2 = coords[3];
            Vector3 scale = new Vector3(x2-x1, 0, y2-y1);
            Vector3 center = new Vector3(x1, 0, y1) + scale/2;
            GameObject roomObj = Instantiate(_roomQuadPrefab, center, Quaternion.identity);
            roomObj.transform.localScale = scale;
        }
    }

}
