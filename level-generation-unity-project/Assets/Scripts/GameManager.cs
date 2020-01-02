using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameManager : MonoBehaviour
{
    private static Player _player;

    private GameManager _instance;

    private void Awake()
    {
        _instance = this;
    }

    public static void RegisterPlayer(Player player)
    {
        _player = player;
    }

    public static Vector3 GetPlayerPosition()
    {
        return _player.transform.position;
    }
}
