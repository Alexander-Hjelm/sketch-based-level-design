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
    [SerializeField] private GameObject _roomQuadFloorPrefab;
    [SerializeField] private GameObject _roomQuadWallPrefab;
    [SerializeField] private GameObject _playerPrefab;
    [SerializeField] private GameObject _enemyPrefab;

    private List<int[]> rectangles = new List<int[]>();
    private List<int[]> circles = new List<int[]>();

    private float _worldScale = 0.25f;

    private bool _playerSpawned = false;

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
            else if(string.Equals(typeStr, "player_spawn"))
            {
                if(!_playerSpawned)
                {
                    // Player spawn
                    Instantiate(_playerPrefab, new Vector3(coords[0], 0.5f, coords[1]), Quaternion.identity);
                    _playerSpawned = true;
                }
                else
                {
                    Debug.LogError("Multiple player spawn points were specified in the map file!");
                }
            }
            else if(string.Equals(typeStr, "enemy_spawn"))
            {
                // Enemy spawn
                Instantiate(_enemyPrefab, new Vector3(coords[0], 0.5f, coords[1]), Quaternion.identity);
            }
        }

        // Create the level grid
        int xSize = (int)(400*_worldScale);
        int ySize = (int)(250*_worldScale);
        int[,] mapGrid = new int[xSize, ySize];

        // Create the level
        foreach(int[] coords in rectangles)
        {
            int x1 = (int)(coords[0]*_worldScale);
            int y1 = (int)(coords[1]*_worldScale);
            int x2 = (int)(coords[2]*_worldScale);
            int y2 = (int)(coords[3]*_worldScale);

            for(int x=x1; x <= x2; x++)
            {
                for(int y=y1; y <= y2; y++)
                {
                    mapGrid[x, y] = 1;
                }
            }
            //Vector3 scale = new Vector3(x2-x1, 0, y2-y1);
            //Vector3 center = new Vector3(x1, 0, y1) + scale/2;
        }

        for(int x=0; x < xSize; x++)
        {
            for(int y=0; y < ySize; y++)
            {
                Vector3 center = new Vector3(x, 0f, y);
                if(mapGrid[x, y] == 1)
                {
                    GameObject roomObj = Instantiate(_roomQuadFloorPrefab, center, Quaternion.identity);
                }
                else
                {
                    GameObject roomObj = Instantiate(_roomQuadWallPrefab, center, Quaternion.identity);
                }
            }
        }
    }

}
